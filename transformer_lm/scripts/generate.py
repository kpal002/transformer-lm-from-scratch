"""Generate text from a trained TransformerLM checkpoint.

Usage:
    python -m transformer_lm.scripts.generate \\
        --checkpoint ./checkpoints/ckpt_final.pt \\
        --vocab ./checkpoints/bpe.vocab \\
        --merges ./checkpoints/bpe.merges \\
        --prompt "Once upon a time"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from transformer_lm.model.transformer import TransformerLM
from transformer_lm.tokenizer import BPETokenizer
from transformer_lm.training import generate


SPECIAL_TOKENS = ["<|endoftext|>"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate text from a TransformerLM checkpoint")
    p.add_argument("--checkpoint", type=Path, required=True, help="Path to .pt checkpoint file")
    p.add_argument("--vocab",      type=Path, required=True, help="BPE vocab file")
    p.add_argument("--merges",     type=Path, required=True, help="BPE merges file")
    p.add_argument("--prompt",     type=str,  default="Once upon a time", help="Text prompt")
    p.add_argument("--max-tokens", type=int,  default=256, help="Maximum tokens to generate")
    p.add_argument("--temperature",type=float,default=0.8)
    p.add_argument("--top-p",      type=float,default=0.95)
    p.add_argument("--num-samples",type=int,  default=1, help="Number of independent completions")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    tokenizer = BPETokenizer.from_files(args.vocab, args.merges, special_tokens=SPECIAL_TOKENS)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model_state = ckpt["model"]
    meta = ckpt.get("meta", {})

    # Infer model shape from checkpoint weights
    embed = model_state["token_embedding.weight"]
    vocab_size, d_model = embed.shape
    num_layers = sum(1 for k in model_state if k.endswith(".attn.W_Q.weight"))

    # Prefer stored metadata; fall back to inferring from weight shapes
    norm_type = meta.get("norm_type", "pre")
    use_rope = meta.get("use_rope", True)

    if "context_length" in meta and "num_heads" in meta:
        context_length = meta["context_length"]
        num_heads = meta["num_heads"]
    elif use_rope:
        rope_cos = model_state[next(k for k in model_state if "cos_cache" in k)]
        context_length, dk = rope_cos.shape
        num_heads = d_model // dk
    else:
        context_length = model_state["pos_embedding.weight"].shape[0]
        num_heads = d_model // 64  # fallback: assume d_k=64

    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        norm_type=norm_type,
        use_rope=use_rope,
    ).to(device)
    model.load_state_dict(model_state)

    prompt_ids = tokenizer.encode(args.prompt)
    eot_id = tokenizer.token_to_id.get("<|endoftext|>".encode())

    for i in range(args.num_samples):
        if args.num_samples > 1:
            print(f"\n── Sample {i+1} ──")
        generated_ids = generate(
            model=model,
            prompt_tokens=prompt_ids,
            context_length=context_length,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            eos_token_id=eot_id,
            device=device,
        )
        print(args.prompt + tokenizer.decode(generated_ids))


if __name__ == "__main__":
    main()
