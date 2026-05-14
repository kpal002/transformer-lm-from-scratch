"""TransformerBlock and full TransformerLM."""

import math
import torch
import torch.nn as nn

from transformer_lm.model.layers import Linear, Embedding, RMSNorm
from transformer_lm.model.ffn import SwiGLUFFN
from transformer_lm.model.attention import CausalMultiHeadSelfAttention


class TransformerBlock(nn.Module):
    """Single Transformer block with pre-norm and residual connections.

    x = x + Attention(RMSNorm(x))
    x = x + FFN(RMSNorm(x))
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float = 10000.0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = CausalMultiHeadSelfAttention(d_model, num_heads, max_seq_len, theta, device, dtype)
        self.norm2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLUFFN(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerLM(nn.Module):
    """Full Transformer language model.

    token_ids -> Embedding -> N x TransformerBlock -> RMSNorm -> Linear -> logits

    Returns unnormalized logits; apply cross-entropy loss externally.
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int | None = None,
        theta: float = 10000.0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if d_ff is None:
            d_ff = math.ceil(int(8 / 3 * d_model) / 64) * 64

        self.token_embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, theta, device, dtype)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """(batch, seq) -> logits (batch, seq, vocab_size)"""
        x = self.token_embedding(token_ids)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.final_norm(x))
