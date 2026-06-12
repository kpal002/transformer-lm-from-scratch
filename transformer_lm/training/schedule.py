import math


def get_lr_cosine_schedule(
    t: int,
    alpha_max: float,
    alpha_min: float,
    T_w: int,
    T_c: int,
) -> float:
    """Cosine annealing with linear warmup (LLaMA / CS336 variant).

    Phase 1 — warmup  [0, T_w):   lr = t/T_w × alpha_max
    Phase 2 — cosine  [T_w, T_c]: lr = alpha_min + 0.5(1 + cos(π·progress)) × (alpha_max - alpha_min)
    Phase 3 — flat    (T_c, ∞):   lr = alpha_min
    """
    if t < T_w:
        return (t / T_w) * alpha_max
    if t > T_c:
        return alpha_min
    progress = (t - T_w) / (T_c - T_w)
    return alpha_min + 0.5 * (1.0 + math.cos(progress * math.pi)) * (alpha_max - alpha_min)
