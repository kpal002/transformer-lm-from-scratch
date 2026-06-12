from pathlib import Path

import numpy as np
import torch


def get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random batch of (input, target) sequences from a token array.

    Args:
        dataset:        1-D numpy array of token IDs (np.memmap works fine).
        batch_size:     Sequences per batch.
        context_length: Tokens per sequence.
        device:         PyTorch device string.

    Returns:
        (inputs, targets) both of shape (batch_size, context_length).
    """
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
    """Tokenize a raw text file and save as a numpy uint16 array.

    Args:
        text_path:   Path to the raw UTF-8 text file.
        tokenizer:   BPETokenizer instance.
        output_path: Where to write the .npy file.
        chunk_size:  Characters per chunk (controls peak memory).

    Returns:
        Path to the saved .npy file.
    """
    from tqdm.auto import tqdm

    text_path = Path(text_path)
    output_path = Path(output_path)

    text = text_path.read_text(encoding="utf-8")
    tokens = tokenizer.encode(text)

    arr = np.array(tokens, dtype=np.uint16)
    np.save(output_path, arr)

    print(f"Saved {len(arr):,} tokens → {output_path}  ({output_path.stat().st_size / 1e6:.1f} MB)")
    return output_path
