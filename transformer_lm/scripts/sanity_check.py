"""Sanity check: GPT-2 Small config forward pass.

Usage:
    python -m transformer_lm.scripts.sanity_check

Verifies output shape (batch=2, seq=128, vocab=50257) and prints parameter count.
"""

import torch
from transformer_lm import TransformerLM

# GPT-2 Small configuration
GPT2_SMALL = dict(
    vocab_size=50_257,
    context_length=1_024,
    d_model=768,
    num_layers=12,
    num_heads=12,
    theta=10_000.0,
)

BATCH_SIZE = 2
SEQ_LEN = 128


def main() -> None:
    print("Building TransformerLM (GPT-2 Small config) ...")
    model = TransformerLM(**GPT2_SMALL)

    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}  (~{total / 1e6:.1f}M; GPT-2 Small is ~117M)\n")

    token_ids = torch.randint(0, GPT2_SMALL["vocab_size"], (BATCH_SIZE, SEQ_LEN))
    print(f"Input  shape: {tuple(token_ids.shape)}")

    with torch.no_grad():
        logits = model(token_ids)

    expected = (BATCH_SIZE, SEQ_LEN, GPT2_SMALL["vocab_size"])
    print(f"Output shape: {tuple(logits.shape)}")
    assert tuple(logits.shape) == expected, f"Expected {expected}, got {tuple(logits.shape)}"
    print("Shape check passed.")


if __name__ == "__main__":
    main()
