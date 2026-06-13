"""Data utilities: random batch sampling and corpus tokenisation.

Training data is stored as a flat numpy uint16 array (a "token stream").
Batches are sampled by choosing random start positions and slicing windows.
"""

from pathlib import Path

import numpy as np
import torch


def get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random batch of (input, target) pairs from a token stream.

    Language modelling objective:
        Given tokens [t0, t1, ..., t_{L-1}], the model predicts the next token
        at every position:
            inputs  = [t0, t1, ..., t_{L-1}]
            targets = [t1, t2, ..., t_L    ]
        i.e. targets are inputs shifted by one position to the right.

    Random windowing:
        A random start index is chosen uniformly from [0, len(dataset) - L - 1]
        for each sequence in the batch.  This gives uncorrelated samples even
        when called many times on a dataset that fits in RAM, at the cost of
        not guaranteeing full corpus coverage per epoch.

    Memory efficiency:
        `dataset` can be a numpy.memmap (memory-mapped file) — only the slices
        that are actually read are loaded into RAM.  This allows training on
        datasets larger than available RAM.

    Args:
        dataset:        1-D numpy array of integer token IDs.  dtype should be
                        uint16 (vocab ≤ 65,535) or int32 for larger vocabularies.
        batch_size:     Number of sequences per batch.
        context_length: Number of tokens per sequence (window size).
        device:         PyTorch device string ("cpu", "cuda", "mps").

    Returns:
        inputs:  (batch_size, context_length) int64 tensor
        targets: (batch_size, context_length) int64 tensor — inputs shifted by 1
    """
    # Leave room for the target window to extend one position past the input
    max_start = len(dataset) - context_length - 1
    starts = np.random.randint(0, max_start, size=(batch_size,))

    inputs  = np.stack([dataset[s     : s + context_length    ] for s in starts])
    targets = np.stack([dataset[s + 1 : s + context_length + 1] for s in starts])

    return (
        torch.tensor(inputs,  dtype=torch.long).to(device),
        torch.tensor(targets, dtype=torch.long).to(device),
    )


def tokenize_and_save(
    text_path: str | Path,
    tokenizer,
    output_path: str | Path,
    chunk_size: int = 100_000,
) -> Path:
    """Tokenize a raw UTF-8 text file and save the token IDs as a numpy array.

    Why save to .npy?
        Repeated tokenisation of large corpora (gigabytes of text) is slow —
        minutes to hours depending on tokenizer speed and corpus size.  Saving
        the tokenized result to disk means subsequent training runs (different
        hyperparameters, resumed runs) skip tokenisation entirely.

    Why uint16?
        Token IDs for vocabularies ≤ 65,535 fit in uint16, halving storage
        compared to int32.  Most BPE vocabularies (8k, 32k, 50k) are within
        this range.  For GPT-4-style 100k vocabularies, switch to uint32.

    Args:
        text_path:   Path to the raw UTF-8 text file to tokenize.
        tokenizer:   A BPETokenizer instance with an .encode(text) method.
        output_path: Where to write the resulting .npy file.
        chunk_size:  Characters per processing chunk (controls peak RAM usage
                     during tokenisation; unused in the current implementation
                     which reads the file in one shot).

    Returns:
        Path to the saved .npy file.
    """
    from tqdm.auto import tqdm

    text_path = Path(text_path)
    output_path = Path(output_path)

    # Read the entire file — for very large corpora consider chunked streaming
    text = text_path.read_text(encoding="utf-8")
    tokens = tokenizer.encode(text)

    arr = np.array(tokens, dtype=np.uint16)
    np.save(output_path, arr)

    print(f"Saved {len(arr):,} tokens → {output_path}  ({output_path.stat().st_size / 1e6:.1f} MB)")
    return output_path
