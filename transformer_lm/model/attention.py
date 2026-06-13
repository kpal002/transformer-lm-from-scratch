"""Attention primitives: softmax, SDPA, RoPE, CausalMultiHeadSelfAttention."""

import math
import torch
import torch.nn as nn
from typing import Optional

from transformer_lm.model.layers import Linear


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Numerically stable softmax: subtracts max before exp to prevent overflow."""
    x = x - x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) @ V

    Args:
        query: (..., n, d_k)
        key:   (..., m, d_k)
        value: (..., m, d_v)
        mask:  (..., n, m) bool — True = attend, False = mask out

    Returns:
        (..., n, d_v)
    """
    d_k = query.shape[-1]
    scores = torch.einsum("...nd,...md->...nm", query, key) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    return torch.einsum("...nm,...mv->...nv", softmax(scores, dim=-1), value)


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE).

    Rotates Q/K in 2D sub-spaces by position-dependent angles:
        θ_i(pos) = pos / base^(2i / d_k)

    Applied only to Q and K — not V. Position encodes where to look, not what to say.
    cos/sin tables are precomputed buffers (not persisted in checkpoints).
    """

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        assert d_k % 2 == 0, "d_k must be even for RoPE"

        k = torch.arange(0, d_k // 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (theta ** (2 * k / d_k))
        positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        angles = torch.cat([torch.outer(positions, inv_freq)] * 2, dim=-1)

        self.register_buffer("cos_cache", angles.cos(), persistent=False)
        self.register_buffer("sin_cache", angles.sin(), persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """[-x2, x1] for input [x1, x2] — the perpendicular rotation component."""
        half = x.shape[-1] // 2
        return torch.cat([-x[..., half:], x[..., :half]], dim=-1)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """Apply RoPE to x at given positions.

        Args:
            x:               (..., seq_len, d_k)
            token_positions: (seq_len,) integer positions
        """
        cos = self.cos_cache[token_positions]
        sin = self.sin_cache[token_positions]
        return x * cos + self._rotate_half(x) * sin


class CausalMultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention with RoPE.

    Splits d_model evenly across num_heads (d_k = d_model // num_heads).
    Each token attends only to itself and earlier positions (causal mask).
    RoPE is applied to Q and K after splitting into heads.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        theta: float = 10000.0,
        use_rope: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_rope = use_rope

        self.W_Q = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_K = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_V = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_O = Linear(d_model, d_model, device=device, dtype=dtype)
        if use_rope:
            self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, seq, d_model) -> (batch, seq, d_model)"""
        batch, seq_len, _ = x.shape

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        Q = split_heads(self.W_Q(x))
        K = split_heads(self.W_K(x))
        V = split_heads(self.W_V(x))

        if self.use_rope:
            positions = torch.arange(seq_len, device=x.device)
            Q = self.rope(Q, positions)
            K = self.rope(K, positions)

        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        out = scaled_dot_product_attention(Q, K, V, mask=causal_mask)

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.W_O(out)
