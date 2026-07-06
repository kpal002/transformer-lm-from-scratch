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
    Z_t = γ_t Z_{t-1} + φ(K_t)(1 - γ_t β_t φ(K_t)ᵀ Z_{t-1})

Output per position:
    out_t = φ(Q_t) @ S_t / (φ(Q_t) · Z_t)

Fast path: when CUDA + Triton are available and d_k × d_v ≤ 16 384 elements
(all standard model presets satisfy this), a fused Triton kernel handles the
full N-step scan in a single CUDA launch per (batch, head) pair.

Slow path: pure-PyTorch sequential loop — correct on CPU/MPS and as a
numerical reference for the kernel.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from transformer_lm.model.gdn_triton import gdn_triton_available, gdn_triton_forward, _MAX_S_ELEMS


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
        Q:     (B, H, N, d_k) — raw projected queries (before feature map)
        K:     (B, H, N, d_k) — raw projected keys
        V:     (B, H, N, d_v)
        gamma: (B, H, N) — decay gates in (0, 1], from sigmoid(W_γ x)
        beta:  (B, H, N) — erase gates in [0, 1], from sigmoid(W_β x)
        eps:   Denominator clamp.

    Returns:
        (B, H, N, d_v)
    """
    phi_Q = F.elu(Q) + 1.0   # (B, H, N, d_k) — positive feature map
    phi_K = F.elu(K) + 1.0

    d_k = Q.shape[-1]
    d_v = V.shape[-1]

    use_triton = (
        gdn_triton_available()
        and Q.is_cuda
        and d_k * d_v <= _MAX_S_ELEMS
    )

    if use_triton:
        return gdn_triton_forward(
            phi_Q.contiguous(),
            phi_K.contiguous(),
            V.contiguous(),
            gamma.contiguous(),
            beta.contiguous(),
            eps=eps,
        )

    return _python_scan(phi_Q, phi_K, V, gamma, beta, eps)


def _python_scan(
    phi_Q: torch.Tensor,
    phi_K: torch.Tensor,
    V: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Pure-PyTorch sequential scan — reference implementation / CPU fallback."""
    B, H, N, d_k = phi_Q.shape
    d_v = V.shape[-1]

    outputs = torch.empty(B, H, N, d_v, device=phi_Q.device, dtype=phi_Q.dtype)
    S = phi_Q.new_zeros(B, H, d_k, d_v)
    Z = phi_Q.new_zeros(B, H, d_k)

    for t in range(N):
        pQ_t = phi_Q[:, :, t]   # (B, H, d_k)
        pK_t = phi_K[:, :, t]
        V_t  = V[:, :, t]       # (B, H, d_v)
        g_t  = gamma[:, :, t]   # (B, H)
        b_t  = beta[:, :, t]    # (B, H)

        # Read what is currently stored at key K_t
        e_S = torch.einsum("bhi,bhij->bhj", pK_t, S)   # φ(K_t)ᵀ S:  (B, H, d_v)
        e_Z = (pK_t * Z).sum(-1)                        # φ(K_t)ᵀ Z:  (B, H)

        gb    = g_t * b_t
        v_eff = V_t  - gb.unsqueeze(-1) * e_S
        z_eff = 1.0  - gb * e_Z

        S = g_t[..., None, None] * S + torch.einsum("bhi,bhj->bhij", pK_t, v_eff)
        Z = g_t.unsqueeze(-1) * Z + pK_t * z_eff.unsqueeze(-1)

        num   = torch.einsum("bhi,bhij->bhj", pQ_t, S)   # (B, H, d_v)
        denom = (pQ_t * Z).sum(-1, keepdim=True)          # (B, H, 1)
        outputs[:, :, t] = num / denom.clamp(min=eps)

    return outputs
