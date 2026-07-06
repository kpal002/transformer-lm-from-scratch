"""Throughput + VRAM benchmark for attention variants.

No dataset required — feeds random token IDs through forward+backward.
Run on each branch to measure tok/sec and peak VRAM.

Usage (run on the appropriate branch for each variant):
    # Naive attention
    python benchmark_attn.py --model-size small --context-length 4096

    # Flash attention
    python benchmark_attn.py --model-size small --context-length 4096 --flash-attention

    # Linear attention  (linear-attention branch)
    python benchmark_attn.py --model-size small --context-length 4096 --linear-attention

    # Gated Delta Net  (gated-delta-net branch)
    python benchmark_attn.py --model-size small --context-length 4096 --gdn-attention

    # Mamba-2  (mamba2-attention branch)
    python benchmark_attn.py --model-size small --context-length 4096 --mamba2-attention

    # Scale sweep — print a table across context lengths
    python benchmark_attn.py --model-size small --sweep
"""

from __future__ import annotations

import argparse
import math
import time
from typing import Optional

import torch
import torch.nn.functional as F

from transformer_lm.model.transformer import TransformerLM
from transformer_lm.training import AdamW


MODEL_PRESETS = {
    "tiny":   dict(d_model=256,  d_ff=768,  num_layers=4,  num_heads=4),
    "small":  dict(d_model=768,  d_ff=3072, num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
}

VOCAB_SIZE = 10_000


def _build_model(arch: dict, context_length: int, use_flash: bool,
                 use_linear: bool, use_gdn: bool, use_mamba2: bool,
                 device: str) -> TransformerLM:
    kwargs: dict = dict(
        vocab_size=VOCAB_SIZE,
        context_length=context_length,
        d_model=arch["d_model"],
        num_layers=arch["num_layers"],
        num_heads=arch["num_heads"],
        d_ff=arch["d_ff"],
    )
    if use_flash:
        kwargs["use_flash"] = True
    # Linear/GDN/Mamba-2 flags exist only on their respective branches;
    # guard with getattr so this script still imports on main.
    import transformer_lm.model.transformer as _tm
    import inspect
    sig = inspect.signature(_tm.TransformerLM.__init__)
    if "use_linear" in sig.parameters and use_linear:
        kwargs["use_linear"] = True
    if "use_gdn" in sig.parameters and use_gdn:
        kwargs["use_gdn"] = True
    if "use_mamba2" in sig.parameters and use_mamba2:
        kwargs["use_mamba2"] = True

    return TransformerLM(**kwargs).to(device)


def _attn_label(args: argparse.Namespace) -> str:
    if getattr(args, "mamba2_attention", False): return "mamba2"
    if getattr(args, "gdn_attention",    False): return "gdn"
    if getattr(args, "linear_attention", False): return "linear"
    if getattr(args, "flash_attention",  False): return "flash"
    return "naive"


def _one_step(model: TransformerLM, x: torch.Tensor) -> torch.Tensor:
    logits = model(x[:, :-1])   # (B, N-1, V)
    return F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), x[:, 1:].reshape(-1))


def run_benchmark(
    model_size: str,
    context_length: int,
    batch_size: int,
    warmup: int,
    steps: int,
    use_flash: bool,
    use_linear: bool,
    use_gdn: bool,
    use_mamba2: bool,
    device: str,
    label: Optional[str] = None,
) -> dict:
    arch = MODEL_PRESETS[model_size]
    model = _build_model(arch, context_length, use_flash, use_linear, use_gdn, use_mamba2, device)

    total_params = sum(p.numel() for p in model.parameters())
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)

    def rand_batch() -> torch.Tensor:
        return torch.randint(0, VOCAB_SIZE, (batch_size, context_length), device=device)

    # Warmup (excluded from timing)
    for _ in range(warmup):
        loss = _one_step(model, rand_batch())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    t0 = time.perf_counter()
    for _ in range(steps):
        loss = _one_step(model, rand_batch())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    if device == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0

    tokens_processed = steps * batch_size * (context_length - 1)
    tok_per_sec = tokens_processed / elapsed
    peak_gb = (torch.cuda.max_memory_allocated(device) / 1e9
               if device == "cuda" else float("nan"))

    return {
        "label":      label or _attn_label(argparse.Namespace(**{
            "flash_attention": use_flash, "linear_attention": use_linear,
            "gdn_attention": use_gdn, "mamba2_attention": use_mamba2,
        })),
        "model_size": model_size,
        "ctx":        context_length,
        "batch":      batch_size,
        "params_M":   total_params / 1e6,
        "tok_per_sec": tok_per_sec,
        "peak_gb":    peak_gb,
        "ms_per_step": elapsed / steps * 1000,
    }


def print_row(r: dict) -> None:
    gb = f"{r['peak_gb']:.1f}" if not math.isnan(r["peak_gb"]) else " n/a"
    print(
        f"  {r['label']:12s}  ctx={r['ctx']:5d}  batch={r['batch']:3d}  "
        f"params={r['params_M']:6.1f}M  "
        f"{r['tok_per_sec']:>10,.0f} tok/s  "
        f"{r['ms_per_step']:>7.1f} ms/step  "
        f"{gb:>5} GB"
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Attention throughput + VRAM benchmark")

    g = p.add_argument_group("model")
    g.add_argument("--model-size", default="small", choices=list(MODEL_PRESETS))
    g.add_argument("--context-length", type=int, default=4096)
    g.add_argument("--batch-size", type=int, default=8)

    g = p.add_argument_group("attention variant")
    g.add_argument("--flash-attention",  dest="flash_attention",  action="store_true")
    g.add_argument("--linear-attention", dest="linear_attention", action="store_true")
    g.add_argument("--gdn-attention",    dest="gdn_attention",    action="store_true")
    g.add_argument("--mamba2-attention", dest="mamba2_attention", action="store_true")

    g = p.add_argument_group("benchmark")
    g.add_argument("--warmup", type=int, default=10, help="Steps excluded from timing")
    g.add_argument("--steps",  type=int, default=100)
    g.add_argument("--sweep",  action="store_true",
                   help="Print throughput table across context lengths 512 1024 2048 4096 8192")

    args = p.parse_args()

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps"  if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}\n")

    if args.sweep:
        print(f"  {'variant':12s}  {'ctx':>7s}  {'batch':>6s}  {'params':>9s}  "
              f"{'tok/sec':>12s}  {'ms/step':>9s}  {'VRAM':>6s}")
        print("  " + "-" * 82)
        for ctx in [512, 1024, 2048, 4096, 8192]:
            r = run_benchmark(
                args.model_size, ctx, args.batch_size,
                args.warmup, args.steps,
                args.flash_attention, args.linear_attention,
                args.gdn_attention, args.mamba2_attention,
                device,
            )
            print_row(r)
    else:
        r = run_benchmark(
            args.model_size, args.context_length, args.batch_size,
            args.warmup, args.steps,
            args.flash_attention, args.linear_attention,
            args.gdn_attention, args.mamba2_attention,
            device,
        )
        print(f"  {'variant':12s}  {'ctx':>7s}  {'batch':>6s}  {'params':>9s}  "
              f"{'tok/sec':>12s}  {'ms/step':>9s}  {'VRAM':>6s}")
        print("  " + "-" * 82)
        print_row(r)


if __name__ == "__main__":
    main()
