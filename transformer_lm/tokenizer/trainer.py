"""BPE trainer with incremental pair-count updates and a max-heap for O(log M) merges.

Key algorithmic choices:
- Doubly-linked list per pre-token → O(1) neighbour lookup during a merge
- Delta patching pair counts → avoids a full O(N) rescan per merge
- Max-heap with lazy deletion → O(log M) best-pair lookup
- Multiprocessing for pre-tokenisation on corpora ≥ 5 MB
"""

from __future__ import annotations

import heapq
import re as _re
from collections import Counter, defaultdict
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Counter as CounterType, Dict, List, Optional, Tuple

import regex as re
from tqdm.auto import tqdm

# GPT-2 pre-tokenisation pattern (handles contractions, words, numbers, punctuation)
_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
_PAT_RE = re.compile(_PAT)
_MP_THRESHOLD_BYTES = 5 * 1024 * 1024


# ---------------------------------------------------------------------------
# Special-token splitting helpers
# ---------------------------------------------------------------------------

def _build_special_split_pattern(special_tokens: List[str]) -> Optional[re.Pattern]:
    if not special_tokens:
        return None
    escaped = sorted((re.escape(t) for t in special_tokens), key=len, reverse=True)
    return re.compile("(" + "|".join(escaped) + ")")


def _split_on_special_tokens(
    text: str,
    pattern: Optional[re.Pattern],
    special_set: frozenset,
) -> List[Tuple[str, bool]]:
    if pattern is None:
        return [(text, False)]
    return [(part, part in special_set) for part in pattern.split(text) if part]


# ---------------------------------------------------------------------------
# Chunk pre-tokenisation (runs in worker processes)
# ---------------------------------------------------------------------------

@dataclass
class _ChunkJob:
    chunk_text: str
    special_tokens: List[str]


def _process_chunk(job: _ChunkJob) -> CounterType:
    pattern = _build_special_split_pattern(job.special_tokens)
    special_set = frozenset(job.special_tokens)
    counts: CounterType = Counter()
    for chunk, is_special in _split_on_special_tokens(job.chunk_text, pattern, special_set):
        if is_special:
            continue
        for m in _PAT_RE.finditer(chunk):
            counts[tuple(bytes([b]) for b in m.group(0).encode("utf-8"))] += 1
    return counts


def _safe_chunk_ranges(text: str, boundary: str, n_chunks: int) -> List[Tuple[int, int]]:
    """Split text into n_chunks ranges aligned to `boundary` token boundaries."""
    total = len(text)
    if total == 0 or n_chunks <= 1 or boundary not in text:
        return [(0, total)]
    approx = max(1, total // n_chunks)
    starts = [0]
    probe = approx
    while probe < total:
        pos = text.find(boundary, probe)
        if pos == -1:
            break
        starts.append(pos)
        probe = pos + len(boundary) + approx
    starts = sorted(set(starts))
    return [(s, starts[i + 1] if i + 1 < len(starts) else total) for i, s in enumerate(starts) if s < (starts[i + 1] if i + 1 < len(starts) else total)]


# ---------------------------------------------------------------------------
# Doubly-linked list for O(1) neighbour access during merge
# ---------------------------------------------------------------------------

@dataclass
class _LinkedSeq:
    vals: list
    prev: list
    next: list
    head: int
    tail: int


def _build_linked_seq(seq: tuple) -> _LinkedSeq:
    n = len(seq)
    vals = [None] + list(seq) + [None]
    prev = list(range(-1, n + 1))
    nxt = list(range(1, n + 3))
    nxt[n + 1] = n + 1
    return _LinkedSeq(vals=vals, prev=prev, next=nxt, head=0, tail=n + 1)


def _linked_pairs(ls: _LinkedSeq):
    i = ls.next[ls.head]
    while ls.next[i] != ls.tail:
        j = ls.next[i]
        yield i, j
        i = j


def _merge_in_linked(ls: _LinkedSeq, a: bytes, b: bytes, ab: bytes) -> list:
    """Merge all (a, b) pairs in ls in-place; return delta list for pair counts."""
    deltas = []
    i = ls.next[ls.head]
    while ls.next[i] != ls.tail:
        j = ls.next[i]
        if ls.vals[i] == a and ls.vals[j] == b:
            li, rj = ls.prev[i], ls.next[j]
            if ls.vals[li] is not None:
                deltas.append(((ls.vals[li], a), -1))
            if ls.vals[rj] is not None:
                deltas.append(((b, ls.vals[rj]), -1))
            ls.vals[i] = ab
            ls.next[i] = rj
            ls.prev[rj] = i
            if ls.vals[li] is not None:
                deltas.append(((ls.vals[li], ab), +1))
            if ls.vals[rj] is not None:
                deltas.append(((ab, ls.vals[rj]), +1))
        else:
            i = j
            continue
        i = ls.next[i]
    return deltas


# ---------------------------------------------------------------------------
# Max-heap with lazy deletion
# ---------------------------------------------------------------------------

class _PairHeap:
    def __init__(self, pair_counts: dict):
        self._counts = dict(pair_counts)
        self._heap = [(-cnt, pair) for pair, cnt in self._counts.items()]
        heapq.heapify(self._heap)

    def update(self, pair: tuple, delta: int) -> None:
        if delta == 0:
            return
        new = self._counts.get(pair, 0) + delta
        if new <= 0:
            self._counts.pop(pair, None)
        else:
            self._counts[pair] = new
            heapq.heappush(self._heap, (-new, pair))

    def best(self) -> Optional[Tuple[tuple, int]]:
        while self._heap:
            neg_cnt, pair = self._heap[0]
            actual = self._counts.get(pair, 0)
            if actual != -neg_cnt:
                heapq.heappop(self._heap)
                continue
            return pair, actual
        return None

    def remove(self, pair: tuple) -> None:
        self._counts.pop(pair, None)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train_bpe(
    input_path: str | Path,
    vocab_size: int,
    special_tokens: Optional[List[str]] = None,
    num_workers: Optional[int] = None,
    show_progress: bool = True,
    log_every: int = 100,
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """Train a BPE tokenizer on a UTF-8 text file.

    Args:
        input_path:     Path to the training corpus.
        vocab_size:     Target vocabulary size (must be ≥ 256 + len(special_tokens)).
        special_tokens: Tokens that are never split (e.g. "<|endoftext|>").
        num_workers:    Worker processes for pre-tokenisation. Defaults to cpu_count()-1.
        show_progress:  Display tqdm progress bars.
        log_every:      Update progress bar postfix every N merges.

    Returns:
        vocab:  Dict mapping token ID → bytes.
        merges: Ordered list of (bytes, bytes) merge pairs.
    """
    special_tokens = special_tokens or []
    min_vocab = 256 + len(special_tokens)
    if vocab_size < min_vocab:
        raise ValueError(f"vocab_size={vocab_size} must be ≥ {min_vocab}")

    text = Path(input_path).read_text(encoding="utf-8")

    # Seed vocabulary: byte tokens 0–255, then special tokens
    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    next_id = 256
    for tok in special_tokens:
        vocab[next_id] = tok.encode("utf-8")
        next_id += 1

    # Pre-tokenise corpus (parallel if corpus is large enough)
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    use_mp = len(text.encode()) >= _MP_THRESHOLD_BYTES and num_workers > 1

    boundary = special_tokens[0] if special_tokens else ""
    ranges = _safe_chunk_ranges(text, boundary, num_workers if use_mp else 1)
    jobs = [_ChunkJob(chunk_text=text[s:e], special_tokens=special_tokens) for s, e in ranges]

    pretoken_counter: CounterType = Counter()
    if use_mp and len(jobs) > 1:
        with Pool(processes=num_workers) as pool:
            it = pool.imap_unordered(_process_chunk, jobs, chunksize=1)
            for partial in (tqdm(it, total=len(jobs), desc="Pre-tokenising", unit="chunk") if show_progress else it):
                pretoken_counter.update(partial)
    else:
        for job in (tqdm(jobs, desc="Pre-tokenising", unit="chunk") if show_progress else jobs):
            pretoken_counter.update(_process_chunk(job))

    # Build linked sequences and initial pair counts
    pretoken_seqs = {seq: (_build_linked_seq(seq), freq) for seq, freq in pretoken_counter.items()}

    pair_counts: defaultdict = defaultdict(int)
    for ls, freq in pretoken_seqs.values():
        for i, j in _linked_pairs(ls):
            pair_counts[(ls.vals[i], ls.vals[j])] += freq

    heap = _PairHeap(pair_counts)
    merges: List[Tuple[bytes, bytes]] = []
    bar = tqdm(total=vocab_size - len(vocab), desc="Learning BPE merges", unit="merge") if show_progress else None

    while len(vocab) < vocab_size:
        result = heap.best()
        if result is None:
            break
        (a, b), best_count = result
        ab = a + b
        heap.remove((a, b))

        vocab[next_id] = ab
        merges.append((a, b))
        next_id += 1

        agg: defaultdict = defaultdict(int)
        for ls, freq in pretoken_seqs.values():
            for delta_pair, delta_val in _merge_in_linked(ls, a, b, ab):
                agg[delta_pair] += delta_val * freq
        agg[(a, b)] -= best_count
        for pair, delta in agg.items():
            if delta:
                heap.update(pair, delta)

        if bar is not None:
            bar.update(1)
            if len(merges) % log_every == 0 or len(vocab) == vocab_size:
                bar.set_postfix(vocab=len(vocab), last=(a + b).decode("utf-8", errors="replace")[:20], count=best_count)

    if bar is not None:
        bar.close()

    return vocab, merges
