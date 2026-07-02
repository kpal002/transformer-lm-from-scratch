"""Causal linear attention (Katharopoulos et al., 2020).

Replaces the O(N²) softmax attention with a linear-time approximation:
  φ(x) = ELU(x) + 1                          # positive feature map
  out_i = (Σ_{j≤i} φ(K_j)⊗V_j) · φ(Q_i)   # causal dot-product
          ─────────────────────────────────
          (Σ_{j≤i} φ(K_j)) · φ(Q_i)

By computing the numerator and denominator as cumulative sums we achieve
O(N·d²) time and O(N·d) memory instead of O(N²·d).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def causal_linear_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Parallel causal linear attention via cumulative outer-product sums.

    Args:
        Q: (batch, heads, seq, d_k)
        K: (batch, heads, seq, d_k)
        V: (batch, heads, seq, d_v)
        eps: denominator clamp to avoid division by zero.

    Returns:
        (batch, heads, seq, d_v)
    """
    phi_Q = F.elu(Q) + 1.0   # (B, H, N, d_k)
    phi_K = F.elu(K) + 1.0   # (B, H, N, d_k)

    # Numerator: cumulative sum of outer products φ(K)⊗V over the seq dim
    # KV[i] = Σ_{j≤i} φ(K_j)^T · V_j   shape: (B, H, N, d_k, d_v)
    KV = torch.einsum("bhnd,bhne->bhnde", phi_K, V)
    cumKV = KV.cumsum(dim=2)  # causal prefix sum

    # Numerator dot: φ(Q_i) · cumKV[i]  → (B, H, N, d_v)
    num = torch.einsum("bhnd,bhnde->bhne", phi_Q, cumKV)

    # Denominator: cumulative sum of φ(K) then dot with φ(Q)
    cumK = phi_K.cumsum(dim=2)          # (B, H, N, d_k)
    denom = (phi_Q * cumK).sum(dim=-1, keepdim=True)  # (B, H, N, 1)

    return num / denom.clamp(min=eps)
