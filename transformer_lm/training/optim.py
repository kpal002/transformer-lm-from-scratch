import torch


class AdamW(torch.optim.Optimizer):
    """AdamW with decoupled weight decay, implemented from scratch."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                state["step"] += 1
                t = state["step"]
                m, v = state["m"], state["v"]

                m.mul_(beta1).add_(g, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)

                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)

                p.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)
                p.add_(p, alpha=-lr * wd)

        return loss


def clip_gradient_norm(parameters, max_norm: float, eps: float = 1e-6) -> float:
    """Clip gradients by global L2 norm. Returns the norm before clipping."""
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return 0.0
    total_norm = torch.sqrt(sum(g.pow(2).sum() for g in grads) + eps)
    if total_norm > max_norm:
        scale = max_norm / total_norm
        for g in grads:
            g.mul_(scale)
    return total_norm.item()
