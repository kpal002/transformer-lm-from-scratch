"""Triton kernel for the Gated Delta Net sequential scan.

Each CUDA block handles one (batch, head) pair.  The full S matrix (d_k × d_v)
and the normaliser vector Z (d_k) live in registers for the entire N-step pass,
eliminating the N separate Python-launched CUDA kernels of the pure-PyTorch loop.

Per-step recurrence (Yang et al. 2024):
    phi_Q, phi_K = ELU(·)+1 feature maps
    e_S  = phi_K_t @ S          — read current value at key K_t
    e_Z  = phi_K_t · Z          — normaliser read
    v_eff = V_t − γβ · e_S      — corrected write value (erase then write)
    z_eff = 1 − γβ · e_Z
    S     ← γ·S + outer(phi_K_t, v_eff)   rank-1 update
    Z     ← γ·Z + phi_K_t · z_eff
    out_t = (phi_Q_t @ S) / max(phi_Q_t · Z, eps)

Register budget: for all standard presets d_k = d_v = 64, so S ≈ 16 KB fp32.
If d_k × d_v > _MAX_S_ELEMS the Python fallback is used (see gated_delta_net.py).

Padding: DK/DV are rounded up to the next power-of-2 for tl.arange sizing;
DK_REAL/DV_REAL carry the true sizes for load/store masks.  Zero-padding in
tensors propagates correctly through the recurrence.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


_MAX_S_ELEMS = 16_384   # 64 KB fp32 — stay within register/L1 budget


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------

if _TRITON_AVAILABLE:

    @triton.jit
    def _gdn_scan_kernel(
        # Tensor pointers (caller guarantees contiguous layout)
        phiQ_ptr, phiK_ptr, V_ptr,
        gamma_ptr, beta_ptr,
        Out_ptr,
        # Strides for (B, H, N, D) tensors
        Q_sb, Q_sh, Q_sn, Q_sd,
        K_sb, K_sh, K_sn, K_sd,
        V_sb, V_sh, V_sn, V_sd,
        g_sb, g_sh, g_sn,        # gamma: (B, H, N)
        b_sb, b_sh, b_sn,        # beta:  (B, H, N)
        O_sb, O_sh, O_sn, O_sd,
        # Runtime dimensions
        H,
        N,
        DK_REAL,                 # actual d_k (≤ DK)
        DV_REAL,                 # actual d_v (≤ DV)
        # Compile-time dimensions (next power-of-2 — one kernel per shape)
        DK: tl.constexpr,
        DV: tl.constexpr,
        eps: tl.constexpr,
    ):
        """One program per (batch, head).  Sequential scan over N positions."""
        bh = tl.program_id(0)
        b  = bh // H
        h  = bh %  H

        q_base  = b * Q_sb  + h * Q_sh
        k_base  = b * K_sb  + h * K_sh
        v_base  = b * V_sb  + h * V_sh
        g_base  = b * g_sb  + h * g_sh
        bt_base = b * b_sb  + h * b_sh
        o_base  = b * O_sb  + h * O_sh

        dk_offs = tl.arange(0, DK)
        dv_offs = tl.arange(0, DV)
        dk_mask = dk_offs < DK_REAL   # zero-pad dk tail when DK > d_k
        dv_mask = dv_offs < DV_REAL   # zero-pad dv tail when DV > d_v

        # Carry state in registers; zero-padded region stays zero throughout
        S = tl.zeros([DK, DV], dtype=tl.float32)
        Z = tl.zeros([DK],     dtype=tl.float32)

        for t in range(N):
            pQ = tl.load(phiQ_ptr + q_base + t * Q_sn + dk_offs * Q_sd,
                         mask=dk_mask, other=0.0).to(tl.float32)
            pK = tl.load(phiK_ptr + k_base + t * K_sn + dk_offs * K_sd,
                         mask=dk_mask, other=0.0).to(tl.float32)
            Vt  = tl.load(V_ptr   + v_base + t * V_sn + dv_offs * V_sd,
                          mask=dv_mask, other=0.0).to(tl.float32)

            g_t = tl.load(gamma_ptr + g_base  + t * g_sn).to(tl.float32)
            b_t = tl.load(beta_ptr  + bt_base + t * b_sn).to(tl.float32)
            gb  = g_t * b_t

            # Read existing content at K_t
            e_S = tl.sum(pK[:, None] * S, axis=0)   # (DV,)
            e_Z = tl.sum(pK * Z)                     # scalar

            # Corrected write value (erase then write)
            v_eff = Vt  - gb * e_S
            z_eff = 1.0 - gb * e_Z

            # Rank-1 state update
            S = g_t * S + pK[:, None] * v_eff[None, :]
            Z = g_t * Z + pK * z_eff

            # Output
            num   = tl.sum(pQ[:, None] * S, axis=0)
            denom = tl.maximum(tl.sum(pQ * Z), eps)

            tl.store(
                Out_ptr + o_base + t * O_sn + dv_offs * O_sd,
                (num / denom).to(Out_ptr.dtype.element_ty),
                mask=dv_mask,
            )


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

def gdn_triton_forward(
    phi_Q: torch.Tensor,
    phi_K: torch.Tensor,
    V:     torch.Tensor,
    gamma: torch.Tensor,
    beta:  torch.Tensor,
    eps:   float = 1e-6,
) -> torch.Tensor:
    """Launch the Triton GDN scan.

    All inputs must be contiguous CUDA tensors.
    phi_Q, phi_K: (B, H, N, d_k) — ELU+1 applied by caller
    V:            (B, H, N, d_v)
    gamma, beta:  (B, H, N)
    """
    B, H, N, d_k = phi_Q.shape
    d_v = V.shape[-1]

    DK = triton.next_power_of_2(d_k)
    DV = triton.next_power_of_2(d_v)

    out = torch.empty(B, H, N, d_v, device=phi_Q.device, dtype=phi_Q.dtype)

    _gdn_scan_kernel[(B * H,)](
        phi_Q, phi_K, V, gamma, beta, out,
        phi_Q.stride(0), phi_Q.stride(1), phi_Q.stride(2), phi_Q.stride(3),
        phi_K.stride(0), phi_K.stride(1), phi_K.stride(2), phi_K.stride(3),
        V.stride(0),     V.stride(1),     V.stride(2),     V.stride(3),
        gamma.stride(0), gamma.stride(1), gamma.stride(2),
        beta.stride(0),  beta.stride(1),  beta.stride(2),
        out.stride(0),   out.stride(1),   out.stride(2),   out.stride(3),
        H=H, N=N,
        DK_REAL=d_k, DV_REAL=d_v,
        DK=DK, DV=DV,
        eps=eps,
        num_warps=4,
    )

    return out


def gdn_triton_available() -> bool:
    return _TRITON_AVAILABLE and torch.cuda.is_available()
