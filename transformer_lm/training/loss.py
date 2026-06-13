"""Cross-entropy loss for next-token prediction, implemented from scratch.

We implement log_softmax manually rather than using F.cross_entropy so that
every numerical step is transparent and inspectable.
"""

import torch


def log_softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Numerically stable log-softmax.

    Naïve implementation:
        log_softmax(x)_i = x_i - log( sum_j exp(x_j) )

    Problem: exp(x_j) overflows for large logits (e.g. x_j > 88 in float32).

    Stable implementation (log-sum-exp trick):
        Subtract max(x) before exponentiating — doesn't change softmax output
        because the constant cancels in numerator and denominator:

        log_softmax(x)_i = (x_i - max(x)) - log( sum_j exp(x_j - max(x)) )

    Args:
        x:   Input tensor of any shape.
        dim: Dimension to normalise over (the vocabulary dimension).

    Returns:
        Log-probabilities of same shape as x, summing to 0 in log-space
        (i.e. probabilities sum to 1).
    """
    x_stable = x - x.max(dim=dim, keepdim=True).values   # subtract max for stability
    log_sum_exp = torch.log(torch.exp(x_stable).sum(dim=dim, keepdim=True))
    return x_stable - log_sum_exp


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy loss for next-token prediction (language modelling).

    Cross-entropy for a single example:
        L = -log P(target | context) = -log_softmax(logits)[target]

    The average over the batch and sequence gives a scalar loss whose
    exponential is the perplexity:  PPL = exp(L).

    Implementation detail — gathering target log-probs:
        Rather than a matrix multiply, we select the log-probability of the
        correct token at each position using integer indexing:
            log_probs_flat[ arange(N), targets_flat ]
        This is equivalent to computing the full softmax and then picking
        the target column, but O(N) rather than O(N * vocab_size) in memory.

    Args:
        logits:  (batch, seq, vocab_size) — raw model outputs (unnormalized).
        targets: (batch, seq) — correct next-token IDs (integers).

    Returns:
        Scalar mean loss.  exp(loss) = perplexity.
    """
    # Convert logits to log-probabilities along the vocabulary dimension
    log_probs = log_softmax(logits, dim=-1)               # (batch, seq, vocab_size)

    # Flatten batch and sequence into a single dimension for indexing
    log_probs_flat = log_probs.view(-1, logits.size(-1))  # (batch*seq, vocab_size)
    targets_flat = targets.view(-1)                        # (batch*seq,)

    # Pick the log-probability of the correct token at each position
    batch_idx = torch.arange(log_probs_flat.size(0), device=logits.device)
    correct_log_probs = log_probs_flat[batch_idx, targets_flat]  # (batch*seq,)

    # Mean negative log-likelihood = cross-entropy loss
    return -correct_log_probs.mean()
