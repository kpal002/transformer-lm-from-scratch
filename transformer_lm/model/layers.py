"""Primitive building blocks: Linear, Embedding, RMSNorm.

All three are implemented from scratch (no nn.Linear, no nn.Embedding) so
every weight, initialisation, and forward pass is fully visible.
"""

import math
import torch
import torch.nn as nn


class Linear(nn.Module):
    """Bias-free linear projection: y = x @ W^T.

    Why no bias?
        Modern LLMs (LLaMA, Mistral, GPT-NeoX) remove biases from all linear
        layers.  With pre-RMSNorm and residual connections the bias term is
        redundant — RMSNorm already re-centres activations — and removing it
        reduces parameter count with no measurable loss impact.

    Weight shape: (out_features, in_features)
        Matches PyTorch's nn.Linear convention.  The forward pass uses einsum
        rather than F.linear so the contraction is explicit and easy to read.

    Initialisation: Glorot (Xavier) normal, truncated at ±3σ
        σ = sqrt(2 / (fan_in + fan_out)) keeps variance roughly constant
        across layers at random init.  Truncation at 3σ removes the rare
        extreme values that can destabilise early training.
    """

    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        sigma = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=sigma, a=-3 * sigma, b=3 * sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project x from in_features to out_features.

        Args:
            x: (..., in_features) — any leading batch/sequence dimensions.

        Returns:
            (..., out_features)
        """
        # '...i,oi->...o' contracts the last dimension of x with the second
        # dimension of weight, equivalent to x @ weight.T but more readable.
        return torch.einsum("...i,oi->...o", x, self.weight)


class Embedding(nn.Module):
    """Token embedding lookup table: maps integer IDs to dense vectors.

    Conceptually a matrix of shape (num_embeddings, embedding_dim) where
    row i is the embedding vector for token i.  The forward pass is a simple
    row-index into the weight matrix — no matrix multiply needed.

    Initialisation: truncated normal (mean=0, std=1)
        Wider than Linear's Glorot init because embedding vectors are summed
        and then fed into a normalised attention layer, so moderate variance
        is fine at initialisation.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Look up embedding vectors for a batch of token IDs.

        Args:
            token_ids: Integer tensor of any shape (...) with values in
                       [0, num_embeddings).

        Returns:
            (..., embedding_dim) — each integer replaced by its row vector.
        """
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    Standard LayerNorm computes:
        y = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias

    RMSNorm simplifies to:
        y = x / RMS(x) * weight     where RMS(x) = sqrt(mean(x²) + eps)

    Why RMSNorm instead of LayerNorm?
        - Removing mean subtraction costs nothing in practice (activations are
          already roughly centred after residual connections).
        - No bias term needed — saves parameters.
        - Slightly faster in practice; used by LLaMA, Mistral, Gemma.

    Numerical stability:
        The division is done in float32 even when the input is bfloat16/float16.
        Mixed-precision training can cause RMS to underflow in 16-bit, producing
        NaN or Inf.  Upcasting to float32 for the normalisation step and then
        casting back preserves precision without the cost of a full float32 pass.

    Learnable scale (weight):
        Initialised to all-ones so the layer starts as an identity.  The model
        learns to scale each feature dimension independently.
    """

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        # Learnable per-dimension scale; shape (d_model,), init to 1.
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise the last dimension of x by its RMS, then scale.

        Args:
            x: (..., d_model)

        Returns:
            (..., d_model) — same shape, normalised.
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)                              # upcast for stability
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return ((x / rms) * self.weight.to(torch.float32)).to(in_dtype)  # cast back
