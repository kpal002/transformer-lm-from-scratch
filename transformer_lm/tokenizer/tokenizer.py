"""BPETokenizer: encode / decode text using a trained BPE vocabulary."""

from __future__ import annotations

import heapq
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import regex as re

# Reuse the same pattern and helper from the trainer
_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
_PAT_RE = re.compile(_PAT)


def _build_special_split_pattern(special_tokens: List[str]) -> Optional[re.Pattern]:
    if not special_tokens:
        return None
    escaped = sorted((re.escape(t) for t in special_tokens), key=len, reverse=True)
    return re.compile("(" + "|".join(escaped) + ")")


def _split_on_special_tokens(text, pattern, special_set):
    if pattern is None:
        return [(text, False)]
    return [(part, part in special_set) for part in pattern.split(text) if part]


class BPETokenizer:
    """BPE tokenizer built from a trained vocabulary and merge list.

    Encoding uses an O(n log n) priority-queue merge and an LRU cache (64k entries)
    so repeated words cost one dict lookup after the first occurrence.
    """

    def __init__(
        self,
        vocab: Dict[int, bytes],
        merges: List[Tuple[bytes, bytes]],
        special_tokens: Optional[List[str]] = None,
    ):
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = special_tokens or []

        self.token_to_id = {tok: tid for tid, tok in self.vocab.items()}

        # Register any special tokens not already in vocab
        next_id = max(self.vocab, default=-1) + 1
        for tok in self.special_tokens:
            tok_bytes = tok.encode("utf-8")
            if tok_bytes not in self.token_to_id:
                self.vocab[next_id] = tok_bytes
                self.token_to_id[tok_bytes] = next_id
                next_id += 1

        self.merge_ranks: Dict[Tuple[bytes, bytes], int] = {
            pair: rank for rank, pair in enumerate(self.merges)
        }
        self._split_pattern = _build_special_split_pattern(self.special_tokens)
        self._special_set = frozenset(self.special_tokens)

        # Bind a per-instance LRU cache so different tokenizer instances stay isolated
        @lru_cache(maxsize=1 << 16)
        def _cached(pretok: str) -> Tuple[int, ...]:
            return tuple(self.token_to_id[p] for p in self._apply_merges(pretok))

        self._cached_encode_pretoken = _cached

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @classmethod
    def from_files(
        cls,
        vocab_path: str | Path,
        merges_path: str | Path,
        special_tokens: Optional[List[str]] = None,
    ) -> "BPETokenizer":
        """Load tokenizer from tab-separated vocab and merges files."""
        vocab: Dict[int, bytes] = {}
        with open(vocab_path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if line:
                    tid, hex_bytes = line.split("\t", 1)
                    vocab[int(tid)] = bytes.fromhex(hex_bytes)

        merges: List[Tuple[bytes, bytes]] = []
        with open(merges_path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if line:
                    left, right = line.split("\t", 1)
                    merges.append((bytes.fromhex(left), bytes.fromhex(right)))

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def save(self, vocab_path: str | Path, merges_path: str | Path) -> None:
        """Save vocab and merges as tab-separated hex files."""
        with open(vocab_path, "w", encoding="utf-8") as f:
            for tid in sorted(self.vocab):
                f.write(f"{tid}\t{self.vocab[tid].hex()}\n")
        with open(merges_path, "w", encoding="utf-8") as f:
            for left, right in self.merges:
                f.write(f"{left.hex()}\t{right.hex()}\n")

    # ------------------------------------------------------------------
    # Core merge logic
    # ------------------------------------------------------------------

    def _apply_merges(self, pretok: str) -> List[bytes]:
        """Apply BPE merges to a single pre-token using a priority queue."""
        seq = [bytes([b]) for b in pretok.encode("utf-8")]
        n = len(seq)
        if n == 1:
            return seq
        if n == 2:
            return [seq[0] + seq[1]] if (seq[0], seq[1]) in self.merge_ranks else seq

        prev = list(range(-1, n))
        nxt = list(range(1, n + 1))
        heap: list = []
        for i in range(n - 1):
            rank = self.merge_ranks.get((seq[i], seq[i + 1]))
            if rank is not None:
                heapq.heappush(heap, (rank, i))

        valid = set(range(n))
        while heap:
            rank, i = heapq.heappop(heap)
            if i not in valid:
                continue
            j = nxt[i]
            if j >= n or j not in valid:
                continue
            if self.merge_ranks.get((seq[i], seq[j])) != rank:
                continue
            seq[i] = seq[i] + seq[j]
            valid.discard(j)
            nxt[i] = nxt[j]
            if nxt[j] < n:
                prev[nxt[j]] = i
            li = prev[i]
            if li >= 0 and li in valid:
                r = self.merge_ranks.get((seq[li], seq[i]))
                if r is not None:
                    heapq.heappush(heap, (r, li))
            ri = nxt[i]
            if ri < n and ri in valid:
                r = self.merge_ranks.get((seq[i], seq[ri]))
                if r is not None:
                    heapq.heappush(heap, (r, i))

        return [seq[i] for i in range(n) if i in valid]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, text: str) -> List[int]:
        """Encode text to a list of token IDs."""
        ids: List[int] = []
        for chunk, is_special in _split_on_special_tokens(text, self._split_pattern, self._special_set):
            if is_special:
                ids.append(self.token_to_id[chunk.encode("utf-8")])
            else:
                for m in _PAT_RE.finditer(chunk):
                    ids.extend(self._cached_encode_pretoken(m.group(0)))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Encode a stream of text chunks, yielding token IDs one at a time."""
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: List[int]) -> str:
        """Decode a list of token IDs back to a string."""
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")
