import torch


def log_softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_stable = x - x.max(dim=dim, keepdim=True).values
    log_sum_exp = torch.log(torch.exp(x_stable).sum(dim=dim, keepdim=True))
    return x_stable - log_sum_exp


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy loss for next-token prediction.

    Args:
        logits:  (batch, seq, vocab_size)
        targets: (batch, seq) — correct next-token IDs

    Returns:
        Scalar mean loss. exp(loss) = perplexity.
    """
    log_probs = log_softmax(logits, dim=-1)
    log_probs_flat = log_probs.view(-1, logits.size(-1))
    targets_flat = targets.view(-1)
    batch_idx = torch.arange(log_probs_flat.size(0), device=logits.device)
    return -log_probs_flat[batch_idx, targets_flat].mean()
