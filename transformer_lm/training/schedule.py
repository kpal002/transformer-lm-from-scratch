"""Learning rate schedule: linear warmup followed by cosine annealing.

This is the schedule used by LLaMA and CS336 — sometimes called the
"cosine with warmup" schedule.
"""

import math


def get_lr_cosine_schedule(
    t: int,
    alpha_max: float,
    alpha_min: float,
    T_w: int,
    T_c: int,
) -> float:
    """Compute the learning rate at step t.

    The schedule has three phases:

    Phase 1 — Linear warmup  [0, T_w):
        lr grows linearly from 0 to alpha_max.
        Why warm up? At random init the model is far from any good solution
        and gradients are noisy.  Starting with a small lr prevents early
        large steps from destabilising training.

            lr(t) = (t / T_w) * alpha_max

    Phase 2 — Cosine annealing  [T_w, T_c]:
        lr decays from alpha_max to alpha_min following a half-cosine curve.
        The cosine shape decays slowly at first (when the model is still
        learning quickly) and accelerates near the end (fine-tuning the
        solution).

            progress = (t - T_w) / (T_c - T_w)   ∈ [0, 1]
            lr(t) = alpha_min + 0.5 * (1 + cos(π * progress)) * (alpha_max - alpha_min)

    Phase 3 — Flat  (T_c, ∞):
        lr stays at alpha_min for any steps beyond T_c.
        This handles the case where training is extended past the scheduled
        horizon without changing the schedule definition.

            lr(t) = alpha_min

    Args:
        t:         Current training step (0-indexed).
        alpha_max: Peak learning rate (reached at end of warmup).
        alpha_min: Minimum learning rate (floor after cosine decay).
        T_w:       Number of warmup steps.
        T_c:       Total cosine decay steps (= total training steps in most runs).

    Returns:
        Learning rate to use at step t.
    """
    if t < T_w:
        # Phase 1: linear ramp from 0 to alpha_max
        return (t / T_w) * alpha_max
    if t > T_c:
        # Phase 3: flat floor
        return alpha_min
    # Phase 2: cosine decay from alpha_max to alpha_min
    progress = (t - T_w) / (T_c - T_w)
    return alpha_min + 0.5 * (1.0 + math.cos(progress * math.pi)) * (alpha_max - alpha_min)
