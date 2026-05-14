"""Download and prepare the TinyStories corpus."""

from pathlib import Path
from typing import Optional

STORY_SEP = "<|endoftext|>"


def download_tinystories(
    output_path: str | Path = "tinystories_train.txt",
    max_stories: Optional[int] = None,
) -> Path:
    """Stream TinyStories from HuggingFace and write a plain UTF-8 corpus file.

    Stories are separated by the `<|endoftext|>` sentinel so the BPE trainer
    can split on special tokens cleanly.

    Args:
        output_path: Where to write the corpus (default: tinystories_train.txt).
        max_stories: Cap on number of stories (None = full ~2.1 M dataset).

    Returns:
        Path to the written corpus file.
    """
    from datasets import load_dataset
    from tqdm.auto import tqdm

    output_path = Path(output_path)
    if output_path.exists():
        print(f"Corpus already exists at {output_path}, skipping download.")
        return output_path

    print("Streaming TinyStories from HuggingFace ...")
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

    written = 0
    with output_path.open("w", encoding="utf-8") as f:
        for example in tqdm(ds, total=max_stories, desc="Writing stories"):
            f.write(example["text"].strip())
            f.write(f"\n{STORY_SEP}\n")
            written += 1
            if max_stories and written >= max_stories:
                break

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"Wrote {written:,} stories → {output_path}  ({size_mb:.1f} MB)")
    return output_path
