"""Benchmark Flash Attention vs naive scaled dot-product attention.

Measures three axes that matter for training:
  1. Wall-clock time (ms) vs sequence length — at what seq_len does Flash win?
  2. Peak GPU memory (MiB) vs sequence length — the O(N) vs O(N²) gap.
  3. Training throughput (tokens/sec) — end-to-end impact on a real model.

Run on a CUDA GPU:

    python -m transformer_lm.scripts.benchmark_attention

Outputs:
    benchmark_time.png    — latency curves (log-log scale to see the crossover)
    benchmark_memory.png  — peak memory curves
    benchmark_throughput.png — tokens/sec bar chart
    benchmark_results.json — raw numbers for further analysis
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Callable

import torch

# Guard: benchmarks only make sense on CUDA
if not torch.cuda.is_available():
    raise SystemExit("CUDA not available — this benchmark requires a GPU.")

from transformer_lm.model.attention import (
    CausalMultiHeadSelfAttention,
    scaled_dot_product_attention,
)
from transformer_lm.model.flash_attention import flash_attention, flash_attention_available
from transformer_lm.model.transformer import TransformerLM

DEVICE = "cuda"


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _reset_peak():
    torch.cuda.reset_peak_memory_stats(DEVICE)


def _peak_mib() -> float:
    return torch.cuda.max_memory_allocated(DEVICE) / 1024 ** 2


def _sync():
    torch.cuda.synchronize(DEVICE)


def timed(fn: Callable, warmup: int = 5, iters: int = 20) -> tuple[float, float]:
    """Return (mean_ms, std_ms) for fn() after warmup."""
    for _ in range(warmup):
        fn()
    _sync()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        _sync()
        times.append((time.perf_counter() - t0) * 1000)

    mean = sum(times) / len(times)
    std = math.sqrt(sum((t - mean) ** 2 for t in times) / len(times))
    return mean, std


# ---------------------------------------------------------------------------
# Benchmark 1: attention kernel time and memory vs seq_len
# ---------------------------------------------------------------------------

def bench_kernel(
    seq_lens: list[int],
    batch: int = 2,
    num_heads: int = 16,
    d_k: int = 64,
) -> dict:
    """Measure naive vs flash attention at the raw QKV level."""
    results: dict = {"seq_lens": seq_lens, "naive_ms": [], "flash_ms": [],
                     "naive_mib": [], "flash_mib": []}

    for seq_len in seq_lens:
        shape = (batch, num_heads, seq_len, d_k)
        Q = torch.randn(shape, dtype=torch.bfloat16, device=DEVICE)
        K = torch.randn(shape, dtype=torch.bfloat16, device=DEVICE)
        V = torch.randn(shape, dtype=torch.bfloat16, device=DEVICE)
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=DEVICE))

        # ── Naive ─────────────────────────────────────────────────────────────
        def naive():
            return scaled_dot_product_attention(Q.float(), K.float(), V.float(), mask=mask)

        _reset_peak()
        ms, _ = timed(naive)
        results["naive_ms"].append(ms)
        results["naive_mib"].append(_peak_mib())

        # Free the score matrix between runs
        torch.cuda.empty_cache()

        # ── Flash ─────────────────────────────────────────────────────────────
        if not flash_attention_available():
            print(f"  [seq={seq_len}] Triton not available — skipping Flash.")
            results["flash_ms"].append(None)
            results["flash_mib"].append(None)
            continue

        def flash():
            return flash_attention(Q, K, V, causal=True)

        _reset_peak()
        ms, _ = timed(flash)
        results["flash_ms"].append(ms)
        results["flash_mib"].append(_peak_mib())

        torch.cuda.empty_cache()

        print(
            f"  seq={seq_len:5d} | naive {results['naive_ms'][-1]:7.2f} ms  "
            f"{results['naive_mib'][-1]:7.1f} MiB  |  "
            f"flash {results['flash_ms'][-1]:7.2f} ms  "
            f"{results['flash_mib'][-1]:7.1f} MiB"
        )

    return results


# ---------------------------------------------------------------------------
# Benchmark 2: end-to-end training throughput (tokens/sec)
# ---------------------------------------------------------------------------

def bench_throughput(
    seq_lens: list[int],
    batch: int = 4,
    d_model: int = 512,
    num_layers: int = 4,
    num_heads: int = 8,   # d_k = d_model / num_heads = 64 — matches bench_kernel default
    num_steps: int = 20,
) -> dict:
    """Measure forward-pass tokens/sec for naive vs flash at the full model level.

    Forward-only (torch.no_grad) keeps the benchmark focused on what Flash
    Attention actually changes — the attention kernel's memory footprint and
    arithmetic intensity.  The backward pass scales proportionally to forward
    (typically ~2×), so training throughput ≈ forward_tps / 3.

    The model runs in float32; the flash kernel internally uses bfloat16 via
    the .to(bfloat16) cast in CausalMultiHeadSelfAttention.forward().
    """
    results: dict = {"seq_lens": seq_lens, "naive_tps": [], "flash_tps": []}

    vocab_size = 1024  # small vocab — we only care about attention cost

    for seq_len in seq_lens:
        for use_flash, tag in [(False, "naive"), (True, "flash")]:
            if use_flash and not flash_attention_available():
                results[f"{tag}_tps"].append(None)
                continue

            # float32 model — the flash kernel handles its own bfloat16 cast internally
            model = TransformerLM(
                vocab_size=vocab_size,
                context_length=seq_len,
                d_model=d_model,
                num_layers=num_layers,
                num_heads=num_heads,
                use_flash=use_flash,
            ).to(DEVICE)
            model.eval()

            @torch.no_grad()
            def step():
                ids = torch.randint(0, vocab_size, (batch, seq_len), device=DEVICE)
                return model(ids)

            try:
                # warmup — catch Triton compile / runtime errors early
                for _ in range(3):
                    step()
                _sync()

                t0 = time.perf_counter()
                for _ in range(num_steps):
                    step()
                _sync()
                elapsed = time.perf_counter() - t0

                tps = batch * seq_len * num_steps / elapsed
                results[f"{tag}_tps"].append(tps)
                print(f"  seq={seq_len:5d} {tag:5s} | {tps:,.0f} tokens/sec")

            except RuntimeError as e:
                print(f"  seq={seq_len:5d} {tag:5s} | FAILED: {e}")
                results[f"{tag}_tps"].append(None)
                # Reset CUDA state so subsequent benchmarks can still run
                torch.cuda.empty_cache()

            finally:
                del model
                torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(kernel: dict, throughput: dict, out_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plots.")
        return

    seq_lens = kernel["seq_lens"]

    # ── Time vs seq_len ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(seq_lens, kernel["naive_ms"], "o-", label="Naive (O(N²))", color="#e15759")
    flash_ms = [v for v in kernel["flash_ms"] if v is not None]
    flash_sl = [seq_lens[i] for i, v in enumerate(kernel["flash_ms"]) if v is not None]
    if flash_ms:
        ax.plot(flash_sl, flash_ms, "s-", label="Flash Attention (O(N))", color="#4e79a7")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Attention kernel latency vs sequence length")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "benchmark_time.png", dpi=150)
    plt.close(fig)

    # ── Memory vs seq_len ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(seq_lens, kernel["naive_mib"], "o-", label="Naive (O(N²))", color="#e15759")
    flash_mib = [v for v in kernel["flash_mib"] if v is not None]
    if flash_mib:
        ax.plot(flash_sl, flash_mib, "s-", label="Flash Attention (O(N))", color="#4e79a7")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Peak GPU memory (MiB)")
    ax.set_title("Attention kernel peak memory vs sequence length")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "benchmark_memory.png", dpi=150)
    plt.close(fig)

    # ── Throughput bar chart ──────────────────────────────────────────────────
    tsl = throughput["seq_lens"]
    naive_tps = throughput["naive_tps"]
    flash_tps = throughput["flash_tps"]

    x = list(range(len(tsl)))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([i - width / 2 for i in x], naive_tps, width, label="Naive", color="#e15759", alpha=0.85)
    valid_flash = [v if v is not None else 0 for v in flash_tps]
    ax.bar([i + width / 2 for i in x], valid_flash, width, label="Flash Attention", color="#4e79a7", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in tsl])
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Forward throughput (tokens/sec)")
    ax.set_title("Forward-pass throughput: naive vs Flash Attention")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "benchmark_throughput.png", dpi=150)
    plt.close(fig)

    print(f"\nPlots saved to {out_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Benchmark Flash vs naive attention")
    p.add_argument("--out-dir", type=Path, default=Path("assets"),
                   help="Directory to write plots and JSON results (default: assets/)")
    p.add_argument("--batch", type=int, default=2, help="Batch size for kernel benchmark")
    p.add_argument("--num-heads", type=int, default=16)
    p.add_argument("--d-k", type=int, default=64, help="Per-head dimension")
    p.add_argument(
        "--seq-lens", type=int, nargs="+",
        default=[128, 256, 512, 1024, 2048, 4096],
        help="Sequence lengths to sweep (default: 128 256 512 1024 2048 4096)",
    )
    p.add_argument(
        "--throughput-seq-lens", type=int, nargs="+",
        default=[256, 512, 1024, 2048],
        help="Sequence lengths for the end-to-end throughput benchmark",
    )
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not flash_attention_available():
        print(
            "WARNING: Triton Flash Attention kernel not available.\n"
            "  Install with: pip install triton\n"
            "  Naive-only numbers will still be collected.\n"
        )

    print("=" * 60)
    print("Benchmark 1: raw attention kernel (time + memory)")
    print("=" * 60)
    kernel = bench_kernel(
        seq_lens=args.seq_lens,
        batch=args.batch,
        num_heads=args.num_heads,
        d_k=args.d_k,
    )

    print()
    print("=" * 60)
    print("Benchmark 2: end-to-end training throughput (tokens/sec)")
    print("=" * 60)
    throughput = bench_throughput(seq_lens=args.throughput_seq_lens)

    results = {"kernel": kernel, "throughput": throughput}
    out_json = args.out_dir / "benchmark_results.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nRaw numbers → {out_json}")

    plot_results(kernel, throughput, args.out_dir)

    # Print summary table
    print()
    print("Summary — kernel speedup (naive / flash latency):")
    print(f"{'seq_len':>8}  {'naive_ms':>10}  {'flash_ms':>10}  {'speedup':>8}  {'mem_ratio':>10}")
    for i, sl in enumerate(kernel["seq_lens"]):
        nm = kernel["naive_ms"][i]
        fm = kernel["flash_ms"][i]
        if fm is not None:
            speedup = nm / fm
            mem_ratio = kernel["naive_mib"][i] / kernel["flash_mib"][i]
            print(f"{sl:>8}  {nm:>10.2f}  {fm:>10.2f}  {speedup:>8.2f}x  {mem_ratio:>10.2f}x")
        else:
            print(f"{sl:>8}  {nm:>10.2f}  {'N/A':>10}  {'N/A':>8}  {'N/A':>10}")


if __name__ == "__main__":
    main()
