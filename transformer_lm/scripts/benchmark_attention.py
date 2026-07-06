"""Benchmark all attention variants: naive, flash, linear, GDN, Mamba-2.

Measures three axes that matter for training:
  1. Wall-clock time (ms) vs sequence length — at what seq_len does each win?
  2. Peak GPU memory (MiB) vs sequence length — the O(N) vs O(N²) gap.
  3. Training throughput (tokens/sec) — end-to-end impact on a real model.

Run on a CUDA GPU:

    python -m transformer_lm.scripts.benchmark_attention

Variants benchmarked:
    naive   — O(N²) scaled dot-product attention (baseline)
    flash   — Flash Attention Triton kernel (O(N) memory, same O(N²) compute)
    linear  — Katharopoulos 2020 cumsum (O(N) compute + memory, no selectivity)
    gdn     — Gated Delta Net (O(N), erase-then-write, Triton kernel)
    mamba2  — Mamba-2 chunked scan (O(N), scalar decay gate)

Outputs:
    benchmark_time.png    — latency curves (log-log scale)
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

# Linear variants — imported with fallbacks so the script runs on any branch
try:
    from transformer_lm.model.linear_attention import causal_linear_attention
    _LINEAR_AVAILABLE = True
except ImportError:
    _LINEAR_AVAILABLE = False

try:
    from transformer_lm.model.gated_delta_net import causal_gated_delta_net
    _GDN_AVAILABLE = True
except ImportError:
    _GDN_AVAILABLE = False

try:
    from transformer_lm.model.mamba2_attention import causal_mamba2_attention
    _MAMBA2_AVAILABLE = True
except ImportError:
    _MAMBA2_AVAILABLE = False

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

def _measure_one(fn, warmup: int = 5, iters: int = 20) -> tuple[float, float]:
    """Run fn(), return (mean_ms, peak_mib) after warmup. Clears cache first."""
    torch.cuda.empty_cache()
    for _ in range(warmup):
        fn()
    _sync()
    _reset_peak()
    ms, _ = timed(fn, warmup=0, iters=iters)
    return ms, _peak_mib()


def bench_kernel(
    seq_lens: list[int],
    batch: int = 2,
    num_heads: int = 16,
    d_k: int = 64,
) -> dict:
    """Measure all attention variants at the raw QKV kernel level."""
    VARIANTS = ["naive", "flash", "linear", "gdn", "mamba2"]
    results: dict = {"seq_lens": seq_lens}
    for v in VARIANTS:
        results[f"{v}_ms"]  = []
        results[f"{v}_mib"] = []

    # Column header
    col = "  {:<8}  {:>5}"
    header = f"\n  {'SEQ':>6}  " + "  ".join(
        f"{'NAIVE':>12}  {'FLASH':>12}  {'LINEAR':>12}  {'GDN':>12}  {'MAMBA2':>12}".split("  ")
    )
    print(f"\n  {'SEQ LEN':>7}  "
          f"{'NAIVE':>14}  {'FLASH':>14}  {'LINEAR':>14}  {'GDN':>14}  {'MAMBA2':>14}")
    print(f"  {'':>7}  "
          f"{'ms / MiB':>14}  {'ms / MiB':>14}  {'ms / MiB':>14}  {'ms / MiB':>14}  {'ms / MiB':>14}")
    print("  " + "-" * 83)

    for seq_len in seq_lens:
        shape = (batch, num_heads, seq_len, d_k)
        Q  = torch.randn(shape, dtype=torch.bfloat16, device=DEVICE)
        K  = torch.randn(shape, dtype=torch.bfloat16, device=DEVICE)
        V  = torch.randn(shape, dtype=torch.bfloat16, device=DEVICE)
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=DEVICE))

        # Gates for GDN / Mamba-2 — per-head scalars in (B, H, N)
        gamma = torch.sigmoid(torch.randn(batch, num_heads, seq_len, device=DEVICE))
        beta  = torch.sigmoid(torch.randn(batch, num_heads, seq_len, device=DEVICE))

        def _cell(ms, mib):
            if ms is None:
                return f"{'OOM':>14}"
            return f"{ms:6.2f}ms {mib:6.0f}MiB"

        row: dict[str, tuple] = {}

        # ── Naive ──────────────────────────────────────────────────────────────
        Qf, Kf, Vf = Q.float(), K.float(), V.float()
        try:
            ms, mib = _measure_one(
                lambda: scaled_dot_product_attention(Qf, Kf, Vf, mask=mask))
        except torch.cuda.OutOfMemoryError:
            ms, mib = None, None
        row["naive"] = (ms, mib)
        results["naive_ms"].append(ms)
        results["naive_mib"].append(mib)
        torch.cuda.empty_cache()

        # ── Flash ──────────────────────────────────────────────────────────────
        if flash_attention_available():
            try:
                ms, mib = _measure_one(lambda: flash_attention(Q, K, V, causal=True))
            except torch.cuda.OutOfMemoryError:
                ms, mib = None, None
        else:
            ms, mib = None, None
        row["flash"] = (ms, mib)
        results["flash_ms"].append(ms)
        results["flash_mib"].append(mib)
        torch.cuda.empty_cache()

        # ── Naive linear (Katharopoulos 2020) ──────────────────────────────────
        if _LINEAR_AVAILABLE:
            try:
                ms, mib = _measure_one(lambda: causal_linear_attention(Qf, Kf, Vf))
            except torch.cuda.OutOfMemoryError:
                ms, mib = None, None
        else:
            ms, mib = None, None
        row["linear"] = (ms, mib)
        results["linear_ms"].append(ms)
        results["linear_mib"].append(mib)
        torch.cuda.empty_cache()

        # ── Gated Delta Net ────────────────────────────────────────────────────
        if _GDN_AVAILABLE:
            try:
                ms, mib = _measure_one(
                    lambda: causal_gated_delta_net(Qf, Kf, Vf, gamma, beta))
            except torch.cuda.OutOfMemoryError:
                ms, mib = None, None
        else:
            ms, mib = None, None
        row["gdn"] = (ms, mib)
        results["gdn_ms"].append(ms)
        results["gdn_mib"].append(mib)
        torch.cuda.empty_cache()

        # ── Mamba-2 chunked scan ───────────────────────────────────────────────
        if _MAMBA2_AVAILABLE:
            try:
                ms, mib = _measure_one(
                    lambda: causal_mamba2_attention(Qf, Kf, Vf, gamma))
            except torch.cuda.OutOfMemoryError:
                ms, mib = None, None
        else:
            ms, mib = None, None
        row["mamba2"] = (ms, mib)
        results["mamba2_ms"].append(ms)
        results["mamba2_mib"].append(mib)
        torch.cuda.empty_cache()

        print(f"  {seq_len:>7}  " + "  ".join(_cell(*row[v]) for v in VARIANTS))

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

_PALETTE = {
    "naive":  "#e15759",
    "flash":  "#4e79a7",
    "linear": "#59a14f",
    "gdn":    "#76b7b2",
    "mamba2": "#f28e2b",
}
_MARKERS = {"naive": "o", "flash": "s", "linear": "^", "gdn": "D", "mamba2": "P"}
_LABELS  = {
    "naive":  "Naive O(N²)",
    "flash":  "Flash (O(N) mem)",
    "linear": "Linear (Katharopoulos)",
    "gdn":    "Gated Delta Net",
    "mamba2": "Mamba-2",
}


def _plot_line(ax, seq_lens, values, variant):
    sl  = [seq_lens[i] for i, v in enumerate(values) if v is not None]
    val = [v for v in values if v is not None]
    if val:
        ax.plot(sl, val,
                _MARKERS[variant] + "-",
                label=_LABELS[variant],
                color=_PALETTE[variant],
                linewidth=1.8, markersize=6)


def plot_results(kernel: dict, throughput: dict, out_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plots.")
        return

    seq_lens = kernel["seq_lens"]
    VARIANTS = ["naive", "flash", "linear", "gdn", "mamba2"]

    # ── Latency vs seq_len ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for v in VARIANTS:
        _plot_line(ax, seq_lens, kernel[f"{v}_ms"], v)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Attention kernel latency vs sequence length")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "benchmark_time.png", dpi=150)
    plt.close(fig)

    # ── Memory vs seq_len ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for v in VARIANTS:
        _plot_line(ax, seq_lens, kernel[f"{v}_mib"], v)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Peak GPU memory (MiB)")
    ax.set_title("Attention kernel peak memory vs sequence length")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "benchmark_memory.png", dpi=150)
    plt.close(fig)

    # ── Throughput bar chart ──────────────────────────────────────────────────
    tsl = throughput["seq_lens"]
    x   = list(range(len(tsl)))
    n_bars = 2   # only naive + flash in throughput benchmark
    width  = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([i - width / 2 for i in x],
           throughput["naive_tps"], width,
           label="Naive", color=_PALETTE["naive"], alpha=0.85)
    valid_flash = [v if v is not None else 0 for v in throughput["flash_tps"]]
    ax.bar([i + width / 2 for i in x],
           valid_flash, width,
           label="Flash Attention", color=_PALETTE["flash"], alpha=0.85)
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

    # ── Summary: speedup and memory savings relative to naive ─────────────────
    VARIANTS = ["flash", "linear", "gdn", "mamba2"]
    print()
    print("Speedup vs naive (naive_ms / variant_ms):")
    hdr = f"  {'SEQ LEN':>7}  " + "  ".join(f"{v.upper():>8}" for v in VARIANTS)
    print(hdr)
    print("  " + "-" * (9 + 11 * len(VARIANTS)))
    for i, sl in enumerate(kernel["seq_lens"]):
        nm = kernel["naive_ms"][i]
        cells = []
        for v in VARIANTS:
            vm = kernel[f"{v}_ms"][i]
            cells.append(f"{nm/vm:>7.1f}x" if (vm and nm) else f"{'N/A':>8}")
        print(f"  {sl:>7}  " + "  ".join(cells))

    print()
    print("Memory savings vs naive (naive_mib / variant_mib):")
    print(hdr)
    print("  " + "-" * (9 + 11 * len(VARIANTS)))
    for i, sl in enumerate(kernel["seq_lens"]):
        nm = kernel["naive_mib"][i]
        cells = []
        for v in VARIANTS:
            vm = kernel[f"{v}_mib"][i]
            cells.append(f"{nm/vm:>7.1f}x" if (vm and nm) else f"{'N/A':>8}")
        print(f"  {sl:>7}  " + "  ".join(cells))


if __name__ == "__main__":
    main()
