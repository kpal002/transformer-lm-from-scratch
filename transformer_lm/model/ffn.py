"""SwiGLU Feed-Forward Network."""

import math
import torch
import torch.nn as nn

from transformer_lm.model.layers import Linear


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward network.

    output = W2( SiLU(W1(x)) * W3(x) )

    Default d_ff = ceil((8/3 * d_model) / 64) * 64 — parameter-equivalent to a
    standard 4× FFN with two matrices, rounded up to the nearest 64 for GPU alignment.
    """

    def __init__(self, d_model: int, d_ff: int | None = None, device=None, dtype=None):
        super().__init__()
        if d_ff is None:
            d_ff = math.ceil(int(8 / 3 * d_model) / 64) * 64

        self.W1 = Linear(d_model, d_ff, device=device, dtype=dtype)  # gate
        self.W2 = Linear(d_ff, d_model, device=device, dtype=dtype)  # output
        self.W3 = Linear(d_model, d_ff, device=device, dtype=dtype)  # values

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.W1(x)
        gate = gate * torch.sigmoid(gate)  # SiLU
        return self.W2(gate * self.W3(x))
