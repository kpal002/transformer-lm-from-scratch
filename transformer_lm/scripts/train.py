"""Train a TransformerLM on TinyStories from the command line.

Typical usage (free Colab T4 or any GPU):

    # 1. Tokenize (once)
    python -m transformer_lm.scripts.train_tokenizer --vocab-size 10000 --output-dir ./run

    # 2. Train
    python -m transformer_lm.scripts.train --out-dir ./run

    # 3. Generate
    python -m transformer_lm.scripts.generate --checkpoint ./run/ckpt_final.pt \\
        --vocab ./run/bpe.vocab --merges ./run/bpe.merges --prompt "Once upon a time"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from transformer_lm.model.transformer import TransformerLM
from transformer_lm.tokenizer import BPETokenizer
from transformer_lm.data import download_tinystories
from transformer_lm.training import AdamW, TrainingConfig, tokenize_and_save, train


SPECIAL_TOKENS = ["<|endoftext|>"]

# Named architecture presets (d_k = d_model / num_heads = 64 in all cases)
MODEL_PRESETS: dict[str, dict] = {
    "small":  dict(d_model=768,  d_ff=3072,  num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096,  num_layers=24, num_heads=16),
    "large":  dict(d_model=1280, d_ff=5120,  num_layers=36, num_heads=20),
    "xl":     dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
    "10b":    dict(d_model=4608, d_ff=12288, num_layers=50, num_heads=36),
}

# Fallback defaults when no preset and no explicit flag
_ARCH_DEFAULTS = dict(d_model=512, num_layers=4, num_heads=16)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a TransformerLM on TinyStories")

    # ── Data ───────────────────────────────────────────────────────────────────
    g = p.add_argument_group("data")
    g.add_argument("--vocab",   type=Path, default=None, help="BPE vocab file (default: <out-dir>/bpe.vocab)")
    g.add_argument("--merges",  type=Path, default=None, help="BPE merges file (default: <out-dir>/bpe.merges)")
    g.add_argument("--train-tokens", type=Path, default=None, help="Pre-tokenized .npy train file")
    g.add_argument("--val-tokens",   type=Path, default=None, help="Pre-tokenized .npy val file")
    g.add_argument("--val-split",    type=float, default=0.05, help="Fraction of stories held out for val (default 0.05)")

    # ── Model ──────────────────────────────────────────────────────────────────
    g = p.add_argument_group("model")
    g.add_argument("--model-size", type=str, default=None,
                   choices=list(MODEL_PRESETS),
                   help="Named architecture preset (small/medium/large/xl/10b). "
                        "Individual flags override the preset.")
    g.add_argument("--vocab-size",     type=int,   default=10_000)
    g.add_argument("--context-length", type=int,   default=256)
    g.add_argument("--d-model",        type=int,   default=None,
                   help="Residual stream width (default 512, overrides --model-size)")
    g.add_argument("--num-layers",     type=int,   default=None,
                   help="Number of transformer blocks (default 4, overrides --model-size)")
    g.add_argument("--num-heads",      type=int,   default=None,
                   help="Attention heads — d_model must be divisible (default 16, overrides --model-size)")
    g.add_argument("--d-ff",           type=int,   default=None,
                   help="FFN width (default: ceil(8/3 × d-model / 64) × 64, overrides --model-size)")
    g.add_argument("--theta",          type=float, default=10_000.0)
    g.add_argument("--norm-type",      type=str,   default="pre", choices=["pre", "post", "none"],
                   help="Normalization placement: pre-norm (default), post-norm, or none")
    g.add_argument("--no-rope",        dest="use_rope", action="store_false",
                   help="Replace RoPE with learned positional embeddings")
    g.add_argument("--flash-attention", dest="flash_attention", action="store_true",
                   help="Use Flash Attention Triton kernel (requires CUDA + triton)")

    # ── Training ───────────────────────────────────────────────────────────────
    g = p.add_argument_group("training")
    g.add_argument("--num-steps",    type=int,   default=5_000)
    g.add_argument("--warmup-steps", type=int,   default=200)
    g.add_argument("--batch-size",   type=int,   default=32)
    g.add_argument("--alpha-max",    type=float, default=1e-3)
    g.add_argument("--alpha-min",    type=float, default=1e-4)
    g.add_argument("--weight-decay", type=float, default=0.1)
    g.add_argument("--max-grad-norm",type=float, default=1.0)
    g.add_argument("--out-dir",      type=Path,  default=Path("checkpoints"))
    g.add_argument("--resume",       type=Path,  default=None, help="Resume from checkpoint path")

    # ── Logging ────────────────────────────────────────────────────────────────
    g = p.add_argument_group("logging")
    g.add_argument("--log-every",  type=int, default=50)
    g.add_argument("--val-every",  type=int, default=500)
    g.add_argument("--save-every", type=int, default=1_000)
    g.add_argument("--wandb",      action="store_true", help="Enable W&B logging")
    g.add_argument("--wandb-project", type=str, default="transformer-lm")
    g.add_argument("--wandb-run",     type=str, default=None)

    return p.parse_args()


def _resolve_arch(args: argparse.Namespace) -> dict:
    """Merge preset → explicit flags → fallback defaults into final arch params."""
    arch = dict(**_ARCH_DEFAULTS)
    if args.model_size:
        arch.update(MODEL_PRESETS[args.model_size])
    # Explicit flags always win over preset
    if args.d_model    is not None: arch["d_model"]    = args.d_model
    if args.num_layers is not None: arch["num_layers"]  = args.num_layers
    if args.num_heads  is not None: arch["num_heads"]   = args.num_heads
    if args.d_ff       is not None: arch["d_ff"]        = args.d_ff
    if arch["d_model"] % arch["num_heads"] != 0:
        raise ValueError(
            f"d_model={arch['d_model']} must be divisible by num_heads={arch['num_heads']}"
        )
    return arch


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    arch = _resolve_arch(args)
    if args.model_size:
        print(f"Model preset: {args.model_size} — "
              f"d_model={arch['d_model']}, num_layers={arch['num_layers']}, "
              f"num_heads={arch['num_heads']}, d_ff={arch.get('d_ff', 'auto')}")

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    # ── Prepare token arrays ───────────────────────────────────────────────────
    train_npy = (args.train_tokens.expanduser() if args.train_tokens else args.out_dir / "train_tokens.npy")
    val_npy   = (args.val_tokens.expanduser()   if args.val_tokens   else args.out_dir / "val_tokens.npy")

    if train_npy.exists() and val_npy.exists():
        print(f"Using pre-tokenized files: {train_npy}, {val_npy}")
    else:
        # Tokenize from scratch — requires tokenizer files
        vocab_path  = args.vocab  or args.out_dir / "bpe.vocab"
        merges_path = args.merges or args.out_dir / "bpe.merges"
        if not vocab_path.exists() or not merges_path.exists():
            raise FileNotFoundError(
                f"Tokenizer files not found: {vocab_path}, {merges_path}\n"
                f"Either run:  python -m transformer_lm.scripts.train_tokenizer "
                f"--vocab-size {args.vocab_size} --output-dir {args.out_dir}\n"
                f"Or pass pre-tokenized files: --train-tokens <path> --val-tokens <path>"
            )
        tokenizer = BPETokenizer.from_files(vocab_path, merges_path, special_tokens=SPECIAL_TOKENS)
        print(f"Tokenizer loaded: vocab_size={tokenizer.vocab_size:,}")

        corpus_path = args.out_dir / "tinystories_train.txt"
        download_tinystories(output_path=corpus_path)

        text = corpus_path.read_text(encoding="utf-8")
        split = int(len(text) * (1 - args.val_split))
        train_text, val_text = text[:split], text[split:]

        print("Tokenizing training split ...")
        tokenize_and_save(train_text if isinstance(train_text, Path) else _write_tmp(train_text, args.out_dir / "_train.txt"), tokenizer, train_npy)
        print("Tokenizing validation split ...")
        tokenize_and_save(val_text   if isinstance(val_text,   Path) else _write_tmp(val_text,   args.out_dir / "_val.txt"),   tokenizer, val_npy)

    train_data = np.load(train_npy, mmap_mode="r")
    val_data   = np.load(val_npy,   mmap_mode="r")
    print(f"Train tokens: {len(train_data):,}  |  Val tokens: {len(val_data):,}")

    # ── Build config ──────────────────────────────────────────────────────────
    cfg = TrainingConfig(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=arch["d_model"],
        num_layers=arch["num_layers"],
        num_heads=arch["num_heads"],
        d_ff=arch.get("d_ff"),
        theta=args.theta,
        norm_type=args.norm_type,
        use_rope=args.use_rope,
        use_flash_attn=args.flash_attention,
        alpha_max=args.alpha_max,
        alpha_min=args.alpha_min,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        num_steps=args.num_steps,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
        log_every=args.log_every,
        val_every=args.val_every,
        save_every=args.save_every,
        out_dir=str(args.out_dir),
        resume=str(args.resume) if args.resume else None,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run,
    )

    # ── Build model ───────────────────────────────────────────────────────────
    model = TransformerLM(
        vocab_size=cfg.vocab_size,
        context_length=cfg.context_length,
        d_model=cfg.d_model,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        theta=cfg.theta,
        norm_type=cfg.norm_type,
        use_rope=cfg.use_rope,
        use_flash=cfg.use_flash_attn,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"Tokens to process: {cfg.batch_size * cfg.context_length * cfg.num_steps:,}")

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.alpha_max,
        betas=(cfg.beta1, cfg.beta2),
        eps=cfg.eps,
        weight_decay=cfg.weight_decay,
    )

    train(model, optimizer, train_data, val_data, cfg, device)


def _write_tmp(text: str, path: Path) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


if __name__ == "__main__":
    main()
