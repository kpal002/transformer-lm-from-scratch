"""SwiGLU Feed-Forward Network (Noam Shazeer, 2020).

Reference: https://arxiv.org/abs/2002.05202
Used in: LLaMA, PaLM, Mistral, and most modern open-source LLMs.
"""

import math
import torch
import torch.nn as nn

from transformer_lm.model.layers import Linear


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward block.

    Standard FFN (GPT-2 style):
        output = W2( GeLU( W1(x) ) )      — 2 matrices, width 4*d_model

    SwiGLU FFN (LLaMA style):
        gate   = SiLU( W1(x) )            — gating signal (activated)
        values =        W3(x)             — value signal (linear)
        output = W2( gate * values )       — element-wise gate then project down

    Why SwiGLU?
        The multiplicative gating lets the network suppress irrelevant
        features dimension-by-dimension before the down-projection.  In
        practice this consistently gives ~0.5 lower perplexity than a
        standard GeLU FFN at the same parameter budget.  See the SwiGLU
        ablation plot in assets/swiglu_vs_silu.png.

    Parameter budget:
        Using three matrices instead of two means d_ff must be smaller than
        4*d_model to keep total parameter count equal.  The standard recipe
        is d_ff = (8/3) * d_model, rounded up to the nearest 64 for GPU
        alignment:

            d_ff = ceil( (8/3 * d_model) / 64 ) * 64

        This keeps the total parameter count (3 * d_model * d_ff) roughly
        equal to a standard two-matrix FFN with d_ff = 4 * d_model
        (2 * d_model * 4*d_model = 8 * d_model²).

    SiLU (Sigmoid Linear Unit):
        SiLU(x) = x * sigmoid(x)
        Smooth, non-monotonic activation that outperforms ReLU and GeLU on
        most language modelling benchmarks.
    """

    def __init__(self, d_model: int, d_ff: int | None = None, device=None, dtype=None):
        """
        Args:
            d_model: Residual stream dimension.
            d_ff:    Inner dimension of the FFN.  Defaults to
                     ceil(8/3 * d_model / 64) * 64 to match LLaMA sizing.
        """
        super().__init__()
        if d_ff is None:
            d_ff = math.ceil(int(8 / 3 * d_model) / 64) * 64

        self.W1 = Linear(d_model, d_ff, device=device, dtype=dtype)  # gate projection
        self.W2 = Linear(d_ff, d_model, device=device, dtype=dtype)  # down projection
        self.W3 = Linear(d_model, d_ff, device=device, dtype=dtype)  # value projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU: output = W2( SiLU(W1(x)) ⊙ W3(x) ).

        Args:
            x: (..., d_model)

        Returns:
            (..., d_model)
        """
        gate = self.W1(x)
        gate = gate * torch.sigmoid(gate)   # SiLU activation on the gate branch
        values = self.W3(x)                 # linear value branch (no activation)
        return self.W2(gate * values)       # element-wise gating then project down
