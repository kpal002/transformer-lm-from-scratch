"""TransformerBlock and full TransformerLM."""

import math
import torch
import torch.nn as nn

from transformer_lm.model.layers import Linear, Embedding, RMSNorm
from transformer_lm.model.ffn import SwiGLUFFN
from transformer_lm.model.attention import CausalMultiHeadSelfAttention


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float = 10000.0,
        norm_type: str = "pre",
        use_rope: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.norm_type = norm_type
        if norm_type != "none":
            self.norm1 = RMSNorm(d_model, device=device, dtype=dtype)
            self.norm2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = CausalMultiHeadSelfAttention(d_model, num_heads, max_seq_len, theta, use_rope, device, dtype)
        self.ffn = SwiGLUFFN(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm_type == "pre":
            x = x + self.attn(self.norm1(x))
            x = x + self.ffn(self.norm2(x))
        elif self.norm_type == "post":
            x = self.norm1(x + self.attn(x))
            x = self.norm2(x + self.ffn(x))
        else:
            x = x + self.attn(x)
            x = x + self.ffn(x)
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
        norm_type: str = "pre",
        use_rope: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if d_ff is None:
            d_ff = math.ceil(int(8 / 3 * d_model) / 64) * 64

        self.use_rope = use_rope
        self.token_embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        if not use_rope:
            self.pos_embedding = Embedding(context_length, d_model, device=device, dtype=dtype)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, theta, norm_type, use_rope, device, dtype)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """(batch, seq) -> logits (batch, seq, vocab_size)"""
        x = self.token_embedding(token_ids)
        if not self.use_rope:
            positions = torch.arange(token_ids.shape[1], device=token_ids.device)
            x = x + self.pos_embedding(positions)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.final_norm(x))
