"""Flash Attention v2 implemented from scratch in Triton.

Paper: "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"
       Tri Dao, 2023. https://arxiv.org/abs/2307.08691

Why Flash Attention?
--------------------
Standard (naive) attention materialises the full (N, N) attention score matrix
in GPU HBM (global memory):

    S = Q @ K^T                 # (N, N)  ← written to HBM
    P = softmax(S)              # (N, N)  ← written to HBM
    O = P @ V                  # (N, d)  ← written to HBM

This is *memory-bandwidth* bound, not compute-bound:
    - Memory: O(N²) for the score matrix
    - HBM reads/writes: O(N²)

Flash Attention avoids materialising S and P by computing attention in tiles
that fit entirely in SRAM (L1 / shared memory):

    For each tile of Q (BLOCK_M rows):
        For each tile of K, V (BLOCK_N rows):
            Compute S_tile = Q_tile @ K_tile^T        # in SRAM
            Online softmax update (m, l, O_acc)       # in SRAM
        Normalise O_acc and write to HBM              # one write per Q tile

Result:
    - Memory: O(N) — only store O, L (logsumexp), no N×N matrix
    - HBM reads: O(N * d) for Q, K, V — each read once
    - 2–4× wall-clock speedup at N=2048+, growing with N

Online softmax (the core trick):
---------------------------------
To compute softmax without the full row of scores in memory, we maintain:
    m_i  = running row-maximum
    l_i  = running row-sum of exp(score - m_i)
    acc  = running output accumulator

When a new tile of scores S_new arrives:
    m_new = max(m_old, max(S_new))
    l_new = exp(m_old - m_new) * l_old  +  sum(exp(S_new - m_new))
    acc   = exp(m_old - m_new) * acc    +  exp(S_new - m_new) @ V_tile

At the end: output = acc / l_new     (exact softmax, computed incrementally)
The stored logsumexp L = m + log(l) is used in the backward pass to
recompute P tiles without storing them.

Kernel grid:
    Forward:  (ceil(N/BLOCK_M), batch*heads)
    Backward dQ:    (ceil(N/BLOCK_M), batch*heads)
    Backward dK/dV: (ceil(N/BLOCK_N), batch*heads)
"""

import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


# ---------------------------------------------------------------------------
# Forward kernel
# ---------------------------------------------------------------------------

if _TRITON_AVAILABLE:

    @triton.jit
    def _flash_fwd_kernel(
        Q, K, V, sm_scale,
        L,          # (Z*H, N_CTX) logsumexp storage for backward
        Out,
        stride_qz, stride_qh, stride_qm, stride_qk,
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vk, stride_vn,
        stride_oz, stride_oh, stride_om, stride_ok,
        Z, H, N_CTX,
        BLOCK_M:      tl.constexpr,
        BLOCK_N:      tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,
        IS_CAUSAL:    tl.constexpr,
    ):
        """
        Each Triton program handles one (batch, head, Q-tile).
        program_id(0) = Q-tile index, program_id(1) = batch*head index.
        """
        start_m = tl.program_id(0)   # which tile of queries we own
        off_hz  = tl.program_id(1)   # flattened (batch, head) index
        off_z   = off_hz // H        # batch index
        off_h   = off_hz % H         # head index

        # Base offsets for this (batch, head) slice
        q_off = off_z * stride_qz + off_h * stride_qh
        k_off = off_z * stride_kz + off_h * stride_kh
        v_off = off_z * stride_vz + off_h * stride_vh
        o_off = off_z * stride_oz + off_h * stride_oh

        # Block pointer to Q tile: rows [start_m*BM, (start_m+1)*BM), cols [0, d_k)
        Q_block_ptr = tl.make_block_ptr(
            base=Q + q_off,
            shape=(N_CTX, BLOCK_DMODEL),
            strides=(stride_qm, stride_qk),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_DMODEL),
            order=(1, 0),
        )
        # K block pointer: transposed (d_k, N_CTX) so we can do Q @ K^T = Q @ K_T
        K_block_ptr = tl.make_block_ptr(
            base=K + k_off,
            shape=(BLOCK_DMODEL, N_CTX),
            strides=(stride_kk, stride_kn),
            offsets=(0, 0),
            block_shape=(BLOCK_DMODEL, BLOCK_N),
            order=(0, 1),
        )
        # V block pointer: rows [0, N_CTX), cols [0, d_k)
        V_block_ptr = tl.make_block_ptr(
            base=V + v_off,
            shape=(N_CTX, BLOCK_DMODEL),
            strides=(stride_vk, stride_vn),
            offsets=(0, 0),
            block_shape=(BLOCK_N, BLOCK_DMODEL),
            order=(1, 0),
        )

        # Initialise online softmax state (one entry per query row in the tile)
        m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)  # running row-max
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)                # running exp-sum
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)  # output accumulator

        # Load Q tile into SRAM — stays resident for the entire inner KV loop
        q = tl.load(Q_block_ptr)

        # Causal: only iterate over KV positions up to the last query in this tile
        hi = (start_m + 1) * BLOCK_M if IS_CAUSAL else N_CTX

        for start_n in range(0, hi, BLOCK_N):
            # Load K (transposed) and V tiles
            k = tl.load(K_block_ptr)
            v = tl.load(V_block_ptr)

            # Attention scores: (BLOCK_M, BLOCK_N)
            qk = tl.dot(q, k) * sm_scale

            # Causal mask: set future positions to a large negative before softmax.
            # We use -1e6 instead of -inf to avoid NaN from exp(-inf - (-inf))
            # when an entire tile row is masked: m_ij would be -inf, and
            # exp(qk - m_ij) = exp(-inf - (-inf)) = exp(NaN) = NaN.
            # With -1e6: m_ij = -1e6, exp(0) = 1 but beta = exp(-1e6 - m_valid) ≈ 0,
            # so the contribution is numerically zero without NaN.
            if IS_CAUSAL:
                offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)  # query positions
                offs_n = start_n          + tl.arange(0, BLOCK_N)   # key positions
                # Allow attending only to positions j ≤ i
                qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, -1e6)

            # Online softmax update for this tile:
            m_ij = tl.max(qk, axis=1)              # (BLOCK_M,) tile row-max
            p    = tl.exp(qk - m_ij[:, None])      # (BLOCK_M, BLOCK_N) un-normalised
            l_ij = tl.sum(p, axis=1)               # (BLOCK_M,) tile row-sum

            # Merge old running stats with new tile stats
            m_i_new = tl.maximum(m_i, m_ij)
            alpha   = tl.exp(m_i   - m_i_new)      # rescale factor for old accumulator
            beta    = tl.exp(m_ij  - m_i_new)      # rescale factor for new tile

            l_i = alpha * l_i + beta * l_ij
            acc = alpha[:, None] * acc + beta[:, None] * tl.dot(p.to(v.dtype), v)
            m_i = m_i_new

            # Advance to next KV tile
            K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
            V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

        # Normalise by the total exp-sum
        acc = acc / l_i[:, None]

        # Store logsumexp = m + log(l): used in backward to recompute P tiles
        l_ptrs = L + off_hz * N_CTX + start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        tl.store(l_ptrs, m_i + tl.log(l_i))

        # Write output tile to HBM (one write per Q tile, not per KV tile)
        Out_block_ptr = tl.make_block_ptr(
            base=Out + o_off,
            shape=(N_CTX, BLOCK_DMODEL),
            strides=(stride_om, stride_ok),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_DMODEL),
            order=(1, 0),
        )
        tl.store(Out_block_ptr, acc.to(Out.type.element_ty))

    # ---------------------------------------------------------------------------
    # Backward kernel 1: compute dQ
    # ---------------------------------------------------------------------------

    @triton.jit
    def _flash_bwd_dq_kernel(
        Q, K, V, sm_scale,
        Out, dOut,
        dQ,
        L, D,       # D[i] = rowsum(O[i] * dO[i]) — precomputed delta
        stride_qz, stride_qh, stride_qm, stride_qk,
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vk, stride_vn,
        Z, H, N_CTX,
        BLOCK_M:      tl.constexpr,
        BLOCK_N:      tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,
        IS_CAUSAL:    tl.constexpr,
    ):
        """Compute dQ for one Q tile by iterating over all K/V tiles."""
        start_m = tl.program_id(0)
        off_hz  = tl.program_id(1)
        off_z   = off_hz // H
        off_h   = off_hz % H

        q_off = off_z * stride_qz + off_h * stride_qh
        k_off = off_z * stride_kz + off_h * stride_kh
        v_off = off_z * stride_vz + off_h * stride_vh

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_DMODEL)

        # Load Q, dO, L, D for this Q tile
        q  = tl.load(Q    + q_off + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)
        do = tl.load(dOut + q_off + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)
        l  = tl.load(L + off_hz * N_CTX + offs_m)  # (BLOCK_M,)
        d  = tl.load(D + off_hz * N_CTX + offs_m)  # (BLOCK_M,) = rowsum(O * dO)

        dq = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

        hi = (start_m + 1) * BLOCK_M if IS_CAUSAL else N_CTX

        for start_n in range(0, hi, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)

            k = tl.load(K + k_off + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk)
            v = tl.load(V + v_off + offs_n[:, None] * stride_vk + offs_d[None, :] * stride_vn)

            # Recompute attention probabilities from stored L
            qk = tl.dot(q, tl.trans(k)) * sm_scale  # (BLOCK_M, BLOCK_N)
            if IS_CAUSAL:
                qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, float('-inf'))
            p = tl.exp(qk - l[:, None])              # (BLOCK_M, BLOCK_N)

            # Softmax backward: dS[i,j] = P[i,j] * (dP[i,j] - D[i])
            dp = tl.dot(do, tl.trans(v))             # dP = dO @ V^T: (BLOCK_M, BLOCK_N)
            ds = p * (dp - d[:, None])               # (BLOCK_M, BLOCK_N)

            # dQ += dS @ K (scaled)
            dq += sm_scale * tl.dot(ds.to(k.dtype), k)

        # Write dQ tile
        tl.store(dQ + q_off + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk, dq)

    # ---------------------------------------------------------------------------
    # Backward kernel 2: compute dK and dV
    # ---------------------------------------------------------------------------

    @triton.jit
    def _flash_bwd_dkdv_kernel(
        Q, K, V, sm_scale,
        Out, dOut,
        dK, dV,
        L, D,
        stride_qz, stride_qh, stride_qm, stride_qk,
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vk, stride_vn,
        Z, H, N_CTX,
        BLOCK_M:      tl.constexpr,
        BLOCK_N:      tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,
        IS_CAUSAL:    tl.constexpr,
    ):
        """Compute dK and dV for one K/V tile by iterating over all Q tiles."""
        start_n = tl.program_id(0)   # which K/V tile we own
        off_hz  = tl.program_id(1)
        off_z   = off_hz // H
        off_h   = off_hz % H

        q_off = off_z * stride_qz + off_h * stride_qh
        k_off = off_z * stride_kz + off_h * stride_kh
        v_off = off_z * stride_vz + off_h * stride_vh

        offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_DMODEL)

        # Load K, V for this tile
        k = tl.load(K + k_off + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk)
        v = tl.load(V + v_off + offs_n[:, None] * stride_vk + offs_d[None, :] * stride_vn)

        dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
        dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)

        # For causal: only Q positions ≥ last position in this K/V tile attend to it.
        # Align lo DOWN to the nearest BLOCK_M boundary: if lo is not BLOCK_M-aligned,
        # the first Q-tile's offs_m would run start_n*BLOCK_N .. start_n*BLOCK_N+BLOCK_M-1,
        # which overflows N_CTX when BLOCK_N < BLOCK_M (64 < 128 in the default config).
        lo = (start_n * BLOCK_N // BLOCK_M) * BLOCK_M if IS_CAUSAL else 0

        for start_m in range(lo, N_CTX, BLOCK_M):
            offs_m = start_m + tl.arange(0, BLOCK_M)

            q  = tl.load(Q    + q_off + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)
            do = tl.load(dOut + q_off + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)
            l  = tl.load(L + off_hz * N_CTX + offs_m)
            d  = tl.load(D + off_hz * N_CTX + offs_m)

            # Recompute P tile
            qk = tl.dot(q, tl.trans(k)) * sm_scale
            if IS_CAUSAL:
                qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, float('-inf'))
            p = tl.exp(qk - l[:, None])   # (BLOCK_M, BLOCK_N)

            # dV += P^T @ dO
            dv += tl.dot(tl.trans(p.to(do.dtype)), do)

            # dK += dS^T @ Q
            dp = tl.dot(do, tl.trans(v))
            ds = p * (dp - d[:, None])
            dk += sm_scale * tl.dot(tl.trans(ds.to(q.dtype)), q)

        tl.store(dK + k_off + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk, dk)
        tl.store(dV + v_off + offs_n[:, None] * stride_vk + offs_d[None, :] * stride_vn, dv)


# ---------------------------------------------------------------------------
# PyTorch autograd Function wrapper
# ---------------------------------------------------------------------------

class FlashAttentionFunction(torch.autograd.Function):
    """Wraps the Triton kernels as a differentiable PyTorch operation."""

    @staticmethod
    def forward(ctx, q, k, v, causal: bool, sm_scale: float):
        """
        Args:
            q, k, v: (batch, heads, seq, d_k) — float16 or bfloat16
            causal:  Apply causal mask (True for language modelling)
            sm_scale: 1 / sqrt(d_k)

        Returns:
            out: (batch, heads, seq, d_k)
        """
        BLOCK_M = 128
        BLOCK_N = 64
        Z, H, N_CTX, d_k = q.shape
        BLOCK_DMODEL = d_k

        assert d_k in (16, 32, 64, 128), f"d_k={d_k} not supported; use 16/32/64/128"
        assert q.dtype in (torch.float16, torch.bfloat16)
        assert q.is_cuda
        # Triton kernel assumes contiguous layout — call .contiguous() before this function
        assert q.is_contiguous(), "q must be contiguous; call q.contiguous() before flash_attention"
        assert k.is_contiguous(), "k must be contiguous; call k.contiguous() before flash_attention"
        assert v.is_contiguous(), "v must be contiguous; call v.contiguous() before flash_attention"

        out = torch.empty_like(q)
        # L stores logsumexp per (batch, head, query_pos) for backward recomputation
        L = torch.empty((Z * H, N_CTX), device=q.device, dtype=torch.float32)

        grid = (triton.cdiv(N_CTX, BLOCK_M), Z * H)
        _flash_fwd_kernel[grid](
            q, k, v, sm_scale,
            L, out,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            Z, H, N_CTX,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_DMODEL,
            IS_CAUSAL=causal,
        )

        ctx.save_for_backward(q, k, v, out, L)
        ctx.grid    = grid
        ctx.causal  = causal
        ctx.sm_scale = sm_scale
        ctx.BLOCK_M  = BLOCK_M
        ctx.BLOCK_N  = BLOCK_N
        ctx.BLOCK_DMODEL = BLOCK_DMODEL
        return out

    @staticmethod
    def backward(ctx, do):
        q, k, v, out, L = ctx.saved_tensors
        Z, H, N_CTX, d_k = q.shape

        do = do.contiguous()

        # Precompute D[i] = rowsum(O[i] * dO[i]) — used in softmax backward
        # D has shape (Z*H, N_CTX)
        D = (out * do).sum(dim=-1).reshape(Z * H, N_CTX)

        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        grid_m = (triton.cdiv(N_CTX, ctx.BLOCK_M), Z * H)
        grid_n = (triton.cdiv(N_CTX, ctx.BLOCK_N), Z * H)

        # Kernel 1: dQ (each program owns a Q tile, iterates over KV)
        _flash_bwd_dq_kernel[grid_m](
            q, k, v, ctx.sm_scale,
            out, do, dq, L, D,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            Z, H, N_CTX,
            BLOCK_M=ctx.BLOCK_M, BLOCK_N=ctx.BLOCK_N, BLOCK_DMODEL=ctx.BLOCK_DMODEL,
            IS_CAUSAL=ctx.causal,
        )

        # Kernel 2: dK, dV (each program owns a KV tile, iterates over Q)
        _flash_bwd_dkdv_kernel[grid_n](
            q, k, v, ctx.sm_scale,
            out, do, dk, dv, L, D,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            Z, H, N_CTX,
            BLOCK_M=ctx.BLOCK_M, BLOCK_N=ctx.BLOCK_N, BLOCK_DMODEL=ctx.BLOCK_DMODEL,
            IS_CAUSAL=ctx.causal,
        )

        return dq, dk, dv, None, None


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
) -> torch.Tensor:
    """Call Flash Attention.

    Args:
        q, k, v: (batch, heads, seq, d_k) — must be float16 or bfloat16 on CUDA.
        causal:  Apply lower-triangular causal mask.

    Returns:
        (batch, heads, seq, d_k) — same shape and dtype as input.
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not installed. Run: pip install triton")
    sm_scale = q.shape[-1] ** -0.5
    return FlashAttentionFunction.apply(q, k, v, causal, sm_scale)


def flash_attention_available() -> bool:
    """Return True if Triton is installed and a CUDA device is available."""
    return _TRITON_AVAILABLE and torch.cuda.is_available()
