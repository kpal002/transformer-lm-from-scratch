"""Causal Mamba-2 attention: linear attention with per-token scalar decay gate.

Mamba-2 (Dao and Gu, 2024) adds a learned scalar gate γ_t ∈ (0,1] to the
linear attention recurrence:

    S_t = γ_t · S_{t-1} + φ(K_t) ⊗ V_t
    Z_t = γ_t · Z_{t-1} + φ(K_t)           (normaliser)
    out_t = φ(Q_t) @ S_t / (φ(Q_t) · Z_t)

γ_t = sigmoid(W_γ x_t) is computed per-head from the current input token.
When γ_t ≈ 1 the model retains long-range context; when γ_t ≈ 0 it resets,
discarding past state before writing the current token.

Implementation — chunked parallel scan (chunk_size=64 default):
    Within each chunk the output is computed via an O(C²) weight matrix
    (analogous to attention scores), avoiding a Python loop over the full N.
    Between chunks, state is propagated sequentially (O(N/C) steps).
    Numerically stable: log-cumproducts stay bounded within each chunk.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def causal_mamba2_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    gamma: torch.Tensor,
    chunk_size: int = 64,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Causal linear attention with per-token scalar decay gate (Mamba-2).

    Args:
        Q:          (B, H, N, d_k)
        K:          (B, H, N, d_k)
        V:          (B, H, N, d_v)
        gamma:      (B, H, N) in (0, 1] — per-head decay gate from input tokens
        chunk_size: Chunk length C for parallel scan. 64 is a good default.
        eps:        Denominator clamp.

    Returns:
        (B, H, N, d_v)
    """
    B, H, N, d_k = Q.shape
    d_v = V.shape[-1]

    phi_Q = F.elu(Q) + 1.0   # (B, H, N, d_k)
    phi_K = F.elu(K) + 1.0   # (B, H, N, d_k)

    outputs = torch.empty(B, H, N, d_v, device=Q.device, dtype=Q.dtype)

    # Carry state: S ∈ ℝ^{d_k × d_v} (value state), Z ∈ ℝ^{d_k} (normaliser)
    S = phi_Q.new_zeros(B, H, d_k, d_v)
    Z = phi_Q.new_zeros(B, H, d_k)

    for s in range(0, N, chunk_size):
        e = min(s + chunk_size, N)
        C = e - s

        pQ = phi_Q[:, :, s:e]   # (B, H, C, d_k)
        pK = phi_K[:, :, s:e]   # (B, H, C, d_k)
        Vc = V[:, :, s:e]       # (B, H, C, d_v)
        gc = gamma[:, :, s:e]   # (B, H, C)

        # log_c[..., t] = Σ_{k=0}^{t} log γ_k  (chunk-local cumulative)
        log_c = torch.log(gc.clamp(min=1e-6)).cumsum(dim=-1)   # (B, H, C)

        # W[b,h,t,j] = exp(log_c[t] - log_c[j])
        #   = decay applied to φ(K_j)'s contribution when read at step t
        #   = 1 when j==t, gc[t] when j==t-1, gc[t]*gc[t-1] when j==t-2, …
        diff = log_c.unsqueeze(-1) - log_c.unsqueeze(-2)       # (B, H, C, C)
        causal_mask = torch.tril(torch.ones(C, C, device=Q.device, dtype=torch.bool))
        W = diff.exp() * causal_mask                           # (B, H, C, C)

        # Contribution of carry state S to each output in this chunk:
        #   decay_start[t] = exp(log_c[t]) = Π_{k=0}^{t} γ_k
        #   num_S[t]   = decay_start[t] * φ(Q_t) @ S
        #   denom_S[t] = decay_start[t] * φ(Q_t) · Z
        decay_start = log_c.exp()                              # (B, H, C)
        num_S   = torch.einsum("bhtk,bhkv->bhtv", pQ, S) * decay_start.unsqueeze(-1)
        denom_S = (pQ * Z.unsqueeze(2)).sum(-1, keepdim=True) * decay_start.unsqueeze(-1)

        # Within-chunk attention-like computation:
        #   QK[t,j]  = φ(Q_t) · φ(K_j)
        #   WQK[t,j] = W[t,j] * QK[t,j]   (masked, decay-weighted scores)
        QK  = torch.einsum("bhtk,bhjk->bhtj", pQ, pK)   # (B, H, C, C)
        WQK = W * QK                                      # (B, H, C, C)

        num_chunk   = torch.einsum("bhtj,bhjv->bhtv", WQK, Vc)   # (B, H, C, d_v)
        denom_chunk = WQK.sum(-1, keepdim=True)                    # (B, H, C, 1)

        outputs[:, :, s:e] = (num_S + num_chunk) / (denom_S + denom_chunk).clamp(min=eps)

        # Advance carry state to end of chunk:
        #   S_after = exp(log_c[-1]) * S + Σ_j exp(log_c[-1] - log_c[j]) * φ(K_j) ⊗ V_j
        #   Z_after = exp(log_c[-1]) * Z + Σ_j exp(log_c[-1] - log_c[j]) * φ(K_j)
        log_c_end   = log_c[:, :, -1:]                  # (B, H, 1)
        decay_total = log_c_end.exp().squeeze(-1)        # (B, H)
        decay_to_end = (log_c_end - log_c).exp()        # (B, H, C)

        pK_scaled = pK * decay_to_end.unsqueeze(-1)     # (B, H, C, d_k)
        S = decay_total[..., None, None] * S + torch.einsum("bhck,bhcv->bhkv", pK_scaled, Vc)
        Z = decay_total.unsqueeze(-1) * Z + pK_scaled.sum(dim=2)

    return outputs
