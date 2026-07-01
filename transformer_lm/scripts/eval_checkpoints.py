"""Evaluate val loss at every checkpoint and write metrics.jsonl.

Usage:
    python -m transformer_lm.scripts.eval_checkpoints \
        --run-dir   run-50k \
        --val-tokens owt_val_50k.npy \
        --vocab-size 50000 \
        --model-size small \
        --context-length 1024
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from transformer_lm.model.transformer import TransformerLM
from transformer_lm.training.data import get_batch
from transformer_lm.training.loss import cross_entropy_loss

SPECIAL_TOKENS = ["<|endoftext|>"]

MODEL_PRESETS: dict[str, dict] = {
    "small":  dict(d_model=768,  d_ff=3072,  num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096,  num_layers=24, num_heads=16),
    "large":  dict(d_model=1280, d_ff=5120,  num_layers=36, num_heads=20),
    "xl":     dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
}


@torch.no_grad()
def estimate_val_loss(model, val_data, batch_size, context_length, device, val_steps=50):
    model.eval()
    total = 0.0
    for _ in range(val_steps):
        inputs, targets = get_batch(val_data, batch_size, context_length, device)
        total += cross_entropy_loss(model(inputs), targets).item()
    return total / val_steps


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir",       type=Path, required=True)
    p.add_argument("--val-tokens",    type=Path, required=True)
    p.add_argument("--vocab-size",    type=int,  required=True)
    p.add_argument("--model-size",    type=str,  default=None, choices=list(MODEL_PRESETS))
    p.add_argument("--context-length",type=int,  default=1024)
    p.add_argument("--batch-size",    type=int,  default=8)
    p.add_argument("--val-steps",     type=int,  default=50)
    p.add_argument("--d-model",       type=int,  default=None)
    p.add_argument("--num-layers",    type=int,  default=None)
    p.add_argument("--num-heads",     type=int,  default=None)
    p.add_argument("--d-ff",          type=int,  default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    arch = dict(d_model=512, num_layers=4, num_heads=8)
    if args.model_size:
        arch.update(MODEL_PRESETS[args.model_size])
    if args.d_model:    arch["d_model"]    = args.d_model
    if args.num_layers: arch["num_layers"] = args.num_layers
    if args.num_heads:  arch["num_heads"]  = args.num_heads
    if args.d_ff:       arch["d_ff"]       = args.d_ff

    val_data = np.load(args.val_tokens.expanduser(), mmap_mode="r")

    # Find all checkpoints, sort by step
    ckpt_paths = sorted(
        glob.glob(str(args.run_dir / "ckpt_*.pt")),
        key=lambda p: int(Path(p).stem.split("_")[1]) if Path(p).stem != "ckpt_final" else 999999,
    )
    if not ckpt_paths:
        raise FileNotFoundError(f"No checkpoints found in {args.run_dir}")

    out_path = args.run_dir / "metrics.jsonl"
    print(f"Writing to {out_path}")

    with open(out_path, "w") as f:
        for ckpt_path in ckpt_paths:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            step = ckpt["step"]

            model = TransformerLM(
                vocab_size=args.vocab_size,
                context_length=args.context_length,
                **arch,
            ).to(device)
            model.load_state_dict(ckpt["model"])

            val_loss = estimate_val_loss(
                model, val_data, args.batch_size, args.context_length, device, args.val_steps
            )
            record = {"step": step, "val_loss": val_loss, "val_ppl": math.exp(val_loss)}
            f.write(json.dumps(record) + "\n")
            f.flush()
            print(f"  step {step:6d} | val_loss {val_loss:.4f} | val_ppl {math.exp(val_loss):.2f}")

            del model
            torch.cuda.empty_cache()

    print("Done.")


if __name__ == "__main__":
    main()
