"""Gated Delta Net: erase-then-write associative memory with global decay.

Yang et al., 2024.  Combines two mechanisms:

  1. Delta rule targeted erase (Widrow-Hoff / Hopfield associative memory):
       Remove whatever is currently stored at key K_t before writing V_t.

       S_t = S_{t-1} - β_t φ(K_t)(φ(K_t)ᵀ S_{t-1}) + φ(K_t) ⊗ V_t

     The term φ(K_t)(φ(K_t)ᵀ S_{t-1}) reads the current value stored at K_t
     and subtracts it — an erase — before the new value is written.  Without
     this, writing the same key twice accumulates rather than replaces, causing
     interference in the associative memory.

  2. Mamba-2 global scalar decay:
       Combine erase with a global forget gate γ_t ∈ (0, 1]:

       S_t = γ_t (S_{t-1} - β_t φ(K_t)(φ(K_t)ᵀ S_{t-1})) + φ(K_t) ⊗ V_t

     This can be rearranged to show the effective write value:

       v_eff_t = V_t - γ_t β_t (φ(K_t)ᵀ S_{t-1})   ←  what gets written at K_t
       S_t = γ_t S_{t-1} + φ(K_t) ⊗ v_eff_t

Both gates are computed per-head from the current input token:
    γ_t = sigmoid(W_γ x_t),  β_t = sigmoid(W_β x_t)

The normaliser state Z follows the same erase-then-write rule:
    Z_t = γ_t (Z_{t-1} - β_t φ(K_t)(φ(K_t)ᵀ Z_{t-1})) + φ(K_t)
        = γ_t Z_{t-1} + φ(K_t)(1 - γ_t β_t φ(K_t)ᵀ Z_{t-1})

Output per position:
    out_t = φ(Q_t) @ S_t / (φ(Q_t) · Z_t)

Sequential scan — the erase step reads the current state S_{t-1}, creating a
data dependency that prevents position-parallel computation.  The scan below
runs N Python iterations; each iteration is a small O(d_k × d_v) GPU kernel.
A Triton kernel would be needed to match flash attention throughput in practice.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def causal_gated_delta_net(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Causal Gated Delta Net: erase-then-write state with scalar decay gate.

    Args:
        Q:     (B, H, N, d_k)
        K:     (B, H, N, d_k)
        V:     (B, H, N, d_v)
        gamma: (B, H, N) — decay gates in (0, 1], from sigmoid(W_γ x)
        beta:  (B, H, N) — erase gates in [0, 1], from sigmoid(W_β x)
        eps:   Denominator clamp.

    Returns:
        (B, H, N, d_v)
    """
    B, H, N, d_k = Q.shape
    d_v = V.shape[-1]

    phi_Q = F.elu(Q) + 1.0   # (B, H, N, d_k) — positive feature map
    phi_K = F.elu(K) + 1.0   # (B, H, N, d_k)

    outputs = torch.empty(B, H, N, d_v, device=Q.device, dtype=Q.dtype)

    # Carry state: S ∈ ℝ^{d_k × d_v} (value state), Z ∈ ℝ^{d_k} (normaliser)
    S = phi_Q.new_zeros(B, H, d_k, d_v)
    Z = phi_Q.new_zeros(B, H, d_k)

    for t in range(N):
        pQ_t = phi_Q[:, :, t]   # (B, H, d_k)
        pK_t = phi_K[:, :, t]   # (B, H, d_k)
        V_t  = V[:, :, t]       # (B, H, d_v)
        g_t  = gamma[:, :, t]   # (B, H)
        b_t  = beta[:, :, t]    # (B, H)

        # Read what is currently stored at key K_t
        e_S = torch.einsum("bhi,bhij->bhj", pK_t, S)   # φ(K_t)ᵀ S:  (B, H, d_v)
        e_Z = (pK_t * Z).sum(-1)                        # φ(K_t)ᵀ Z:  (B, H)

        # Effective write: V_t corrected by erasing the old content at K_t.
        # v_eff = V_t - γ_t β_t (what was at K_t)
        gb    = g_t * b_t                                     # (B, H)
        v_eff = V_t  - gb.unsqueeze(-1) * e_S                 # (B, H, d_v)
        z_eff = 1.0  - gb * e_Z                               # (B, H)

        # State update: decay old state + write corrected value at K_t
        S = g_t[..., None, None] * S + torch.einsum("bhi,bhj->bhij", pK_t, v_eff)
        Z = g_t.unsqueeze(-1) * Z + pK_t * z_eff.unsqueeze(-1)

        # Output for position t
        num   = torch.einsum("bhi,bhij->bhj", pQ_t, S)   # (B, H, d_v)
        denom = (pQ_t * Z).sum(-1, keepdim=True)          # (B, H, 1)
        outputs[:, :, t] = num / denom.clamp(min=eps)

    return outputs
