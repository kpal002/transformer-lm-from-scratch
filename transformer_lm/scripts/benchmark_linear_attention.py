"""Benchmark naive vs flash vs linear attention: wall-clock time and peak memory.

Sweeps over sequence lengths [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
with a fixed batch×heads×d_k configuration and reports:
  - forward-pass time (ms)
  - peak GPU memory (MB)

Usage:
    python -m transformer_lm.scripts.benchmark_linear_attention
    python -m transformer_lm.scripts.benchmark_linear_attention --heads 16 --d-k 64 --batch 2
"""

from __future__ import annotations

import argparse
import math
import time

import torch

from transformer_lm.model.attention import scaled_dot_product_attention
from transformer_lm.model.linear_attention import causal_linear_attention

try:
    from transformer_lm.model.flash_attention import flash_attention, flash_attention_available
    HAS_FLASH = flash_attention_available()
except Exception:
    HAS_FLASH = False


def benchmark_fn(fn, *args, warmup: int = 3, repeats: int = 10):
    """Return (mean_ms, peak_mem_mb) for a callable on CUDA."""
    device = next(a.device for a in args if isinstance(a, torch.Tensor))
    for _ in range(warmup):
        fn(*args)
        torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn(*args)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000 / repeats
    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return elapsed_ms, peak_mb


def run_naive(Q, K, V):
    seq = Q.shape[2]
    mask = torch.tril(torch.ones(seq, seq, dtype=torch.bool, device=Q.device))
    return scaled_dot_product_attention(Q.float(), K.float(), V.float(), mask=mask)


def run_flash(Q, K, V):
    return flash_attention(
        Q.contiguous().to(torch.bfloat16),
        K.contiguous().to(torch.bfloat16),
        V.contiguous().to(torch.bfloat16),
        causal=True,
    )


def run_linear(Q, K, V):
    return causal_linear_attention(Q.float(), K.float(), V.float())


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch",    type=int, default=2)
    p.add_argument("--heads",    type=int, default=16)
    p.add_argument("--d-k",      type=int, default=64)
    p.add_argument("--warmup",   type=int, default=3)
    p.add_argument("--repeats",  type=int, default=10)
    p.add_argument("--seq-lens", type=int, nargs="+",
                   default=[256, 512, 1024, 2048, 4096, 8192, 16384, 32768])
    p.add_argument("--no-flash", action="store_true", help="Skip flash attention")
    return p.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        print("CUDA not available — aborting benchmark.")
        return

    device = "cuda"
    B, H, D = args.batch, args.heads, args.d_k

    variants = [("naive", run_naive)]
    if HAS_FLASH and not args.no_flash:
        variants.append(("flash", run_flash))
    variants.append(("linear", run_linear))

    header = f"{'seq':>8}" + "".join(f"  {name:>8}_ms  {name:>8}_MB" for name, _ in variants)
    print(header)
    print("-" * len(header))

    for seq in args.seq_lens:
        Q = torch.randn(B, H, seq, D, device=device)
        K = torch.randn(B, H, seq, D, device=device)
        V = torch.randn(B, H, seq, D, device=device)

        row = f"{seq:>8}"
        for name, fn in variants:
            try:
                ms, mb = benchmark_fn(fn, Q, K, V, warmup=args.warmup, repeats=args.repeats)
                row += f"  {ms:>10.2f}  {mb:>10.1f}"
            except torch.cuda.OutOfMemoryError:
                row += f"  {'OOM':>10}  {'OOM':>10}"
                torch.cuda.empty_cache()
        print(row)


if __name__ == "__main__":
    main()
