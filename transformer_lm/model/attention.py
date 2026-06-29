"""Attention primitives: softmax, scaled dot-product attention, RoPE, and
causal multi-head self-attention.

All operations are implemented from scratch so every line is inspectable.
"""

import math
import torch
import torch.nn as nn
from typing import Optional

from transformer_lm.model.layers import Linear
from transformer_lm.model.flash_attention import flash_attention, flash_attention_available


# ---------------------------------------------------------------------------
# Softmax
# ---------------------------------------------------------------------------

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Numerically stable softmax.

    Naïve exp(x) / sum(exp(x)) overflows for large x.  Subtracting the max
    before exponentiation is mathematically equivalent but keeps values in a
    safe range:

        softmax(x)_i = exp(x_i - max(x)) / sum_j exp(x_j - max(x))

    Args:
        x:   Input tensor of any shape.
        dim: Dimension to normalise over.

    Returns:
        Tensor of same shape as x, summing to 1 along `dim`.
    """
    x = x - x.max(dim=dim, keepdim=True).values  # shift for numerical stability
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


# ---------------------------------------------------------------------------
# Scaled dot-product attention
# ---------------------------------------------------------------------------

def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Attention(Q, K, V) = softmax( QK^T / sqrt(d_k) ) @ V.

    The 1/sqrt(d_k) scaling prevents dot products from growing large when d_k
    is big, which would push softmax into its saturated (near-zero gradient)
    region.

    Args:
        query: (..., n, d_k) — query vectors for n positions.
        key:   (..., m, d_k) — key vectors for m positions.
        value: (..., m, d_v) — value vectors for m positions.
        mask:  (..., n, m) boolean — True means "attend", False means "ignore".
               Masked positions receive -inf before softmax → probability 0.

    Returns:
        (..., n, d_v) — weighted sum of values for each query position.
    """
    d_k = query.shape[-1]
    # Compute raw attention scores: (..., n, m)
    scores = torch.einsum("...nd,...md->...nm", query, key) / math.sqrt(d_k)
    if mask is not None:
        # Replace masked positions with -inf so they become 0 after softmax
        scores = scores.masked_fill(~mask, float("-inf"))
    # Softmax over the key dimension, then weighted sum of values
    return torch.einsum("...nm,...mv->...nv", softmax(scores, dim=-1), value)


# ---------------------------------------------------------------------------
# Rotary Positional Embedding (RoPE)
# ---------------------------------------------------------------------------

class RotaryPositionalEmbedding(nn.Module):
    """Rotary Positional Embedding (Su et al., 2021 — RoFormer).

    Reference: https://arxiv.org/abs/2104.09864
    Used in: LLaMA, Mistral, GPT-NeoX, and most modern open-source LLMs.

    Core idea:
        Instead of adding position information to token embeddings (absolute
        PE) or computing relative biases in the attention matrix (relative PE),
        RoPE rotates Q and K vectors in 2D sub-spaces by position-dependent
        angles.  This encodes *relative* positions implicitly:

            Q_pos · K_pos' = f(pos - pos')

        i.e. the dot product between a query at position `pos` and a key at
        position `pos'` depends only on the *difference* (pos - pos').

    Rotation mechanics:
        The d_k-dimensional vector is split into d_k/2 pairs.  Each pair (x1, x2)
        at position `pos` is rotated by angle θ_i(pos):

            θ_i(pos) = pos / base^(2i / d_k)    for i = 0, 1, ..., d_k/2 - 1

        The rotation is computed via the identity:
            [x1, x2] rotated by θ  →  [x1·cos θ − x2·sin θ,  x2·cos θ + x1·sin θ]

        In matrix form for the full d_k vector:
            RoPE(x, pos) = x ⊙ cos(θ(pos)) + rotate_half(x) ⊙ sin(θ(pos))

        where rotate_half([x1, x2, ...]) = [-x2, x1, ...] — the perpendicular
        component of each 2D pair.

    Why only Q and K, not V?
        Position encodes *where to attend*, not *what the attended value means*.
        Applying RoPE to V would distort the output representation.

    Pre-computed buffers:
        cos and sin tables of shape (max_seq_len, d_k) are registered as
        non-persistent buffers (not saved to checkpoints) since they can always
        be recomputed from theta and d_k.

    base (theta) parameter:
        Controls the frequency of the slowest-rotating dimension.  Default
        10,000 (original paper).  LLaMA 3 uses 500,000 to extend context.
    """

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        assert d_k % 2 == 0, "d_k must be even — RoPE rotates pairs of dimensions"

        # Compute inverse frequencies: shape (d_k/2,)
        # θ_i = 1 / base^(2i / d_k)  →  lower i = higher frequency (changes fast with pos)
        k = torch.arange(0, d_k // 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (theta ** (2 * k / d_k))

        # Outer product: positions × inv_freq → shape (max_seq_len, d_k/2)
        # Then concatenate twice to cover all d_k dims: (max_seq_len, d_k)
        positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        angles = torch.cat([torch.outer(positions, inv_freq)] * 2, dim=-1)

        # Register as non-persistent buffers: moved with .to(device) but not saved
        self.register_buffer("cos_cache", angles.cos(), persistent=False)
        self.register_buffer("sin_cache", angles.sin(), persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the perpendicular rotation component.

        For input [x1, x2, x3, x4, ...] (length d_k):
            Returns [-x_{d_k/2+1}, ..., -x_{d_k}, x1, ..., x_{d_k/2}]

        When multiplied by sin and added to x*cos, this implements 2D rotation
        for each pair (x_i, x_{i + d_k/2}).
        """
        half = x.shape[-1] // 2
        return torch.cat([-x[..., half:], x[..., :half]], dim=-1)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """Apply RoPE rotation to x at the given sequence positions.

        Args:
            x:               (..., seq_len, d_k) — Q or K after head splitting.
            token_positions: (seq_len,) integer positions, usually 0..seq_len-1.

        Returns:
            (..., seq_len, d_k) — rotationally encoded tensor.
        """
        cos = self.cos_cache[token_positions]  # (seq_len, d_k)
        sin = self.sin_cache[token_positions]  # (seq_len, d_k)
        return x * cos + self._rotate_half(x) * sin


# ---------------------------------------------------------------------------
# Causal multi-head self-attention
# ---------------------------------------------------------------------------

class CausalMultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention with optional RoPE.

    Multi-head attention (Vaswani et al., 2017):
        Each head independently attends over the sequence with a d_k = d_model/h
        dimensional sub-space.  The outputs of all h heads are concatenated and
        projected back to d_model.  Multiple heads let the model attend to
        different aspects of context simultaneously.

    Causal mask:
        Each position can only attend to itself and earlier positions (lower
        triangular mask).  This ensures the language model cannot "see the
        future" during training.

    RoPE (optional, default=True):
        When enabled, Q and K are rotated by position-dependent angles before
        the dot product.  When disabled (--no-rope), no positional information
        is injected into attention — a learned position embedding table in
        TransformerLM compensates at the input level.

    Parameter layout:
        W_Q, W_K, W_V: (d_model, d_model) — packed projections for all heads.
        W_O:           (d_model, d_model) — output projection.
        Total: 4 × d_model² parameters (same as nn.MultiheadAttention).

    Head splitting:
        The packed projections are reshaped into (batch, num_heads, seq, d_k)
        so each head sees its own d_k-dimensional slice.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        theta: float = 10000.0,
        use_rope: bool = True,
        use_flash: bool = False,
        device=None,
        dtype=None,
    ):
        """
        Args:
            d_model:     Residual stream dimension.  Must be divisible by num_heads.
            num_heads:   Number of attention heads.
            max_seq_len: Maximum sequence length (needed to pre-compute RoPE tables).
            theta:       RoPE base frequency.  10,000 is the original value;
                         larger values (e.g. 500,000) extend the effective context.
            use_rope:    Whether to apply RoPE to Q and K.  Set False for the
                         no-rope ablation (learned positional embeddings are used
                         at the TransformerLM level instead).
            use_flash:   Use Flash Attention Triton kernel instead of naive SDPA.
                         Requires CUDA + Triton.  Falls back to naive if unavailable.
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # per-head dimension
        self.use_rope = use_rope
        # Use flash attention only if explicitly requested and hardware supports it
        self.use_flash = use_flash and flash_attention_available()

        # Packed QKV projections and output projection — all bias-free
        self.W_Q = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_K = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_V = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_O = Linear(d_model, d_model, device=device, dtype=dtype)

        # Only instantiate RoPE buffers when needed
        if use_rope:
            self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run causal multi-head self-attention on the input sequence.

        Args:
            x: (batch, seq_len, d_model)

        Returns:
            (batch, seq_len, d_model) — attended and projected output.
        """
        batch, seq_len, _ = x.shape

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            """Reshape (batch, seq, d_model) → (batch, num_heads, seq, d_k)."""
            return t.view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Project to Q, K, V and split into per-head slices
        Q = split_heads(self.W_Q(x))  # (batch, num_heads, seq, d_k)
        K = split_heads(self.W_K(x))
        V = split_heads(self.W_V(x))

        # Apply RoPE to Q and K so dot products encode relative positions
        if self.use_rope:
            positions = torch.arange(seq_len, device=x.device)
            Q = self.rope(Q, positions)
            K = self.rope(K, positions)

        if self.use_flash:
            # Flash Attention: O(N) memory, fused kernel — requires float16/bfloat16
            # Must be contiguous: split_heads() uses transpose() which produces a
            # non-contiguous view; our Triton kernel assumes contiguous strides.
            orig_dtype = Q.dtype
            Q_h = Q.contiguous().to(torch.bfloat16)
            K_h = K.contiguous().to(torch.bfloat16)
            V_h = V.contiguous().to(torch.bfloat16)
            out = flash_attention(Q_h, K_h, V_h, causal=True).to(orig_dtype)
        else:
            # Naive attention: materialises the full (seq, seq) score matrix in HBM
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
            out = scaled_dot_product_attention(Q, K, V, mask=causal_mask)

        # Merge heads back: (batch, seq, d_model)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)

        # Final output projection
        return self.W_O(out)
