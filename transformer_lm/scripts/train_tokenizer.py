"""Train a BPE tokenizer on TinyStories.

Usage:
    python -m transformer_lm.scripts.train_tokenizer [--vocab-size N] [--max-stories N]
                                                      [--output-dir DIR]

Steps:
    1. Download TinyStories corpus (skipped if file already exists)
    2. Train BPE with `train_bpe`
    3. Save vocab + merges files
    4. Quick encode/decode round-trip to verify correctness
"""

import argparse
import time
from pathlib import Path

from transformer_lm.data import download_tinystories
from transformer_lm.tokenizer import BPETokenizer, train_bpe


SPECIAL_TOKENS = ["<|endoftext|>"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a BPE tokenizer on TinyStories")
    p.add_argument("--vocab-size", type=int, default=10_000)
    p.add_argument("--max-stories", type=int, default=None, help="Cap stories (None = full dataset)")
    p.add_argument("--output-dir", type=Path, default=Path("."), help="Where to save vocab/merges files")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Data ────────────────────────────────────────────────────────
    corpus_path = args.output_dir / "tinystories_train.txt"
    download_tinystories(output_path=corpus_path, max_stories=args.max_stories)

    # ── 2. Train ───────────────────────────────────────────────────────
    print(f"\nTraining BPE  vocab_size={args.vocab_size:,} ...")
    t0 = time.time()
    vocab, merges = train_bpe(
        input_path=corpus_path,
        vocab_size=args.vocab_size,
        special_tokens=SPECIAL_TOKENS,
    )
    elapsed = time.time() - t0
    print(f"Training complete in {elapsed:.1f}s  |  vocab={len(vocab):,}  merges={len(merges):,}")

    # ── 3. Save ────────────────────────────────────────────────────────
    vocab_path = args.output_dir / "bpe.vocab"
    merges_path = args.output_dir / "bpe.merges"
    tokenizer = BPETokenizer(vocab, merges, special_tokens=SPECIAL_TOKENS)
    tokenizer.save(vocab_path, merges_path)
    print(f"Saved → {vocab_path}  |  {merges_path}")

    # ── 4. Verify ──────────────────────────────────────────────────────
    tokenizer = BPETokenizer.from_files(vocab_path, merges_path, special_tokens=SPECIAL_TOKENS)
    sample = "Once upon a time, there was a little girl named Lily. <|endoftext|>"
    ids = tokenizer.encode(sample)
    decoded = tokenizer.decode(ids)
    tokens = [tokenizer.vocab[i].decode("utf-8", errors="replace") for i in ids]

    print("\nRound-trip check:")
    print(f"  Input   : {sample!r}")
    print(f"  Tokens  : {tokens}")
    print(f"  IDs     : {ids}")
    print(f"  Decoded : {decoded!r}")
    assert decoded == sample, "Round-trip mismatch!"
    print("  ✓ OK")

    # ── 5. Stats ───────────────────────────────────────────────────────
    bench = "Once upon a time, there was a little girl named Lily who loved to explore."
    raw_bytes = len(bench.encode("utf-8"))
    n_tokens = len(tokenizer.encode(bench))
    print(f"\nCompression on benchmark sentence:")
    print(f"  {raw_bytes} bytes → {n_tokens} tokens  ({raw_bytes / n_tokens:.2f} bytes/token)")


if __name__ == "__main__":
    main()
