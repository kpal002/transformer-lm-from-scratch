"""AdamW optimizer and gradient clipping, implemented from scratch.

AdamW reference: Loshchilov & Hutter (2019) — "Decoupled Weight Decay
Regularization"  https://arxiv.org/abs/1711.05101
"""

import torch


class AdamW(torch.optim.Optimizer):
    """AdamW with decoupled weight decay, implemented from scratch.

    Adam (Adaptive Moment Estimation) maintains a running estimate of the
    first and second moments of the gradient for each parameter:

        m_t = β1 * m_{t-1} + (1 - β1) * g_t          # first moment  (mean)
        v_t = β2 * v_{t-1} + (1 - β2) * g_t²          # second moment (variance)

    Bias correction compensates for the zero initialisation of m and v:
        m̂_t = m_t / (1 - β1^t)
        v̂_t = v_t / (1 - β2^t)

    Parameter update (Adam part):
        θ_t = θ_{t-1} - lr * m̂_t / (sqrt(v̂_t) + ε)

    Decoupled weight decay ("W" in AdamW):
        Standard Adam applies weight decay through the gradient (L2 reg), which
        interacts badly with the adaptive scaling.  AdamW applies it directly:

        θ_t = θ_{t-1} - lr * wd * θ_{t-1}      ← separate from gradient step

        This means weight decay shrinks parameters proportionally to their
        current magnitude, independent of gradient scale.

    Typical hyperparameters for transformer LMs:
        lr:           1e-3 (peak, with cosine decay)
        betas:        (0.9, 0.999)
        eps:          1e-8
        weight_decay: 0.1  (aggressive but standard for LLMs)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform one AdamW update step.

        Args:
            closure: Optional callable that re-evaluates the loss (rarely used
                     in practice for LM training).

        Returns:
            Loss value if closure is provided, else None.
        """
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue  # skip frozen parameters

                g = p.grad  # current gradient

                # Lazily initialise optimizer state on first step
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)   # first moment (EMA of g)
                    state["v"] = torch.zeros_like(p)   # second moment (EMA of g²)

                state["step"] += 1
                t = state["step"]
                m, v = state["m"], state["v"]

                # Update biased moment estimates (in-place for memory efficiency)
                m.mul_(beta1).add_(g, alpha=1 - beta1)          # m = β1*m + (1-β1)*g
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)   # v = β2*v + (1-β2)*g²

                # Bias-corrected estimates
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)

                # Gradient step: θ -= lr * m̂ / (√v̂ + ε)
                p.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)

                # Decoupled weight decay: θ -= lr * wd * θ
                p.add_(p, alpha=-lr * wd)

        return loss


def clip_gradient_norm(parameters, max_norm: float, eps: float = 1e-6) -> float:
    """Clip all parameter gradients by the global L2 norm.

    Why clip gradients?
        Without clipping, a single large-loss batch can produce enormous
        gradients that cause the parameters to jump to a completely different
        region of the loss landscape ("exploding gradients").  Clipping the
        global norm to max_norm rescales *all* gradients proportionally when
        the norm exceeds the threshold, preserving the gradient direction.

    Global L2 norm:
        total_norm = sqrt( sum over all params of sum(g²) )

    Scaling when norm > max_norm:
        g ← g * (max_norm / total_norm)

    Why global (not per-parameter)?
        Per-parameter clipping distorts the relative direction of the gradient
        vector across parameters.  Global clipping only rescales the magnitude.

    Args:
        parameters: Iterable of nn.Parameter (typically model.parameters()).
        max_norm:   Gradient norm threshold (typical: 1.0).
        eps:        Small constant added under the sqrt to avoid division by
                    zero when gradients are exactly zero.

    Returns:
        The gradient norm *before* clipping (useful for logging to W&B).
    """
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return 0.0

    # Compute global L2 norm across all parameter tensors
    total_norm = torch.sqrt(sum(g.pow(2).sum() for g in grads) + eps)

    if total_norm > max_norm:
        scale = max_norm / total_norm
        for g in grads:
            g.mul_(scale)

    return total_norm.item()
