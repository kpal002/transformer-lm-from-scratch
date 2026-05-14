"""Primitive building blocks: Linear, Embedding, RMSNorm."""

import math
import torch
import torch.nn as nn


class Linear(nn.Module):
    """Bias-free linear layer.

    Weight shape: (out_features, in_features), matching nn.Linear convention.
    Initialized with Glorot normal, truncated at ±3σ.
    """

    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        sigma = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=sigma, a=-3 * sigma, b=3 * sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(..., in_features) -> (..., out_features)"""
        return torch.einsum("...i,oi->...o", x, self.weight)


class Embedding(nn.Module):
    """Token embedding lookup table.

    Stores a (num_embeddings, embedding_dim) matrix.
    Initialized with truncated normal (mean=0, std=1).
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Integer tensor (...) -> embedding vectors (..., embedding_dim)"""
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Normalizes by RMS (no mean subtraction), then applies a learned scale.
    Computation is done in float32 for stability, then cast back to input dtype.
    """

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return ((x / rms) * self.weight.to(torch.float32)).to(in_dtype)
