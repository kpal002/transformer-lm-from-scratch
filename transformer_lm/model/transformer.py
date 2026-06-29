"""TransformerBlock and full TransformerLM (decoder-only, LLaMA-style).

Architecture overview:
    token_ids
        → Embedding (+ learned pos embedding if use_rope=False)
        → N × TransformerBlock
        → RMSNorm
        → Linear (lm_head)
        → logits (batch, seq, vocab_size)

Each TransformerBlock follows the pre-norm residual pattern used by LLaMA:
    x = x + Attention( RMSNorm(x) )
    x = x + FFN(      RMSNorm(x) )

The norm_type flag lets you ablate to post-norm (original "Attention Is All
You Need" style) or no-norm (diverges without careful lr tuning).
"""

import math
import torch
import torch.nn as nn

from transformer_lm.model.layers import Linear, Embedding, RMSNorm
from transformer_lm.model.ffn import SwiGLUFFN
from transformer_lm.model.attention import CausalMultiHeadSelfAttention


class TransformerBlock(nn.Module):
    """Single transformer layer: attention + FFN with residual connections.

    Supports three normalisation placements via norm_type:

    "pre"  (default, LLaMA-style):
        x = x + Attention( RMSNorm(x) )
        x = x + FFN(       RMSNorm(x) )
        Advantage: gradient flows cleanly through residuals even at depth.
        Most stable; the modern standard.

    "post" (original Transformer, Vaswani 2017):
        x = RMSNorm( x + Attention(x) )
        x = RMSNorm( x + FFN(x) )
        Requires careful warm-up; can diverge at high learning rates.
        See assets/norm_ablation.png for empirical evidence.

    "none" (no normalisation):
        x = x + Attention(x)
        x = x + FFN(x)
        Almost always diverges at lr=1e-3; needs lr ≤ 1e-4.
        Included as an ablation only.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float = 10000.0,
        norm_type: str = "pre",
        use_rope: bool = True,
        use_flash: bool = False,
        device=None,
        dtype=None,
    ):
        """
        Args:
            d_model:     Residual stream width.
            num_heads:   Number of attention heads (d_model must be divisible).
            d_ff:        FFN inner dimension.
            max_seq_len: Max sequence length (for RoPE pre-computation).
            theta:       RoPE base frequency.
            norm_type:   "pre" | "post" | "none"  — where normalisation sits.
            use_rope:    If False, attention has no positional encoding;
                         position is handled by a learned table in TransformerLM.
            use_flash:   Use Flash Attention Triton kernel (requires CUDA + Triton).
        """
        super().__init__()
        self.norm_type = norm_type

        # Only allocate norm layers when they'll actually be used
        if norm_type != "none":
            self.norm1 = RMSNorm(d_model, device=device, dtype=dtype)
            self.norm2 = RMSNorm(d_model, device=device, dtype=dtype)

        self.attn = CausalMultiHeadSelfAttention(
            d_model, num_heads, max_seq_len, theta, use_rope, use_flash, device, dtype
        )
        self.ffn = SwiGLUFFN(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attention + FFN with residuals.

        Args:
            x: (batch, seq_len, d_model)

        Returns:
            (batch, seq_len, d_model)
        """
        if self.norm_type == "pre":
            # Normalise BEFORE each sub-layer → most stable for deep networks
            x = x + self.attn(self.norm1(x))
            x = x + self.ffn(self.norm2(x))
        elif self.norm_type == "post":
            # Normalise AFTER each sub-layer → original Transformer style
            x = self.norm1(x + self.attn(x))
            x = self.norm2(x + self.ffn(x))
        else:
            # No normalisation — ablation only; typically diverges at default lr
            x = x + self.attn(x)
            x = x + self.ffn(x)
        return x


class TransformerLM(nn.Module):
    """Full decoder-only language model (LLaMA / GPT style).

    Produces unnormalized logits over the vocabulary for every position in the
    input sequence.  Apply cross-entropy loss externally (see training/loss.py).

    Positional encoding strategy:
        RoPE (use_rope=True, default):
            No positional table at the input level.  Position is injected into
            Q and K inside each attention layer via rotation.  This is the
            modern standard: positions generalise better and there's no upper
            bound set at model construction time beyond max_seq_len.

        Learned absolute embeddings (use_rope=False):
            A trainable table of shape (context_length, d_model) is added to
            the token embeddings at the input.  Classic GPT-2 style; works
            well within the training context window but degrades on longer
            sequences at inference.

    Final norm:
        A single RMSNorm before the lm_head keeps logit magnitudes stable
        regardless of depth.  This layer is present regardless of norm_type.
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
        use_flash: bool = False,
        device=None,
        dtype=None,
    ):
        """
        Args:
            vocab_size:      Number of tokens in the vocabulary.
            context_length:  Maximum sequence length the model can process.
                             Also used as the positional embedding table size
                             when use_rope=False.
            d_model:         Width of the residual stream throughout the model.
            num_layers:      Depth — number of TransformerBlocks stacked.
            num_heads:       Attention heads per block (d_model % num_heads == 0).
            d_ff:            FFN inner width.  Defaults to
                             ceil(8/3 * d_model / 64) * 64 (LLaMA sizing).
            theta:           RoPE base frequency (default 10,000).
            norm_type:       "pre" | "post" | "none" — passed to every block.
            use_rope:        If False, learned positional embeddings are used
                             instead of RoPE.
        """
        super().__init__()
        if d_ff is None:
            d_ff = math.ceil(int(8 / 3 * d_model) / 64) * 64

        self.use_rope = use_rope

        # Token embedding: maps integer IDs → dense vectors of shape d_model
        self.token_embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)

        # Learned positional embedding table — only used when RoPE is disabled
        if not use_rope:
            self.pos_embedding = Embedding(context_length, d_model, device=device, dtype=dtype)

        # Stack of transformer blocks
        self.use_flash = use_flash
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model, num_heads, d_ff, context_length,
                theta, norm_type, use_rope, use_flash, device, dtype
            )
            for _ in range(num_layers)
        ])

        # Final normalisation before the vocabulary projection
        self.final_norm = RMSNorm(d_model, device=device, dtype=dtype)

        # Un-tied language model head: projects d_model → vocab_size logits
        # (weight tying to token_embedding is a common memory saving trick but
        #  is omitted here to keep the code explicit)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Compute next-token logits for every position in the input.

        Args:
            token_ids: (batch, seq_len) — integer token IDs in [0, vocab_size).

        Returns:
            (batch, seq_len, vocab_size) — unnormalized logits.
            Position i predicts the token at position i+1 (standard LM shift).
        """
        # Embed tokens into d_model-dimensional vectors
        x = self.token_embedding(token_ids)

        # Add positional information when not using RoPE
        if not self.use_rope:
            positions = torch.arange(token_ids.shape[1], device=token_ids.device)
            x = x + self.pos_embedding(positions)

        # Pass through each transformer block
        for block in self.blocks:
            x = block(x)

        # Normalise and project to vocabulary logits
        return self.lm_head(self.final_norm(x))
