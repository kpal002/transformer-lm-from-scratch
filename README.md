# LLM Scaling Laws from Scratch

> I didn't just build a transformer — I ran the Chinchilla isoFLOP experiments from scratch and fit my own scaling laws. My data says the compute-optimal split is **N_opt ∝ C^0.225**, not C^0.50. Here's everything, in pure PyTorch, no black boxes.

![IsoFLOP scaling law curves](assets/isoflop_scaling_laws.png)

*Three compute budgets (10^16–2×10^17 FLOPs), 8 training runs, power-law fits for N_opt and D_opt.*

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kpal002/transformer-lm-from-scratch/blob/main/transformer_lm/notebooks/transformer_from_scratch.ipynb)
![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)
![License MIT](https://img.shields.io/badge/License-MIT-green)

---

## What's inside

Everything is hand-coded — no HuggingFace, no `trl`, no shortcuts.

| Component | What's from scratch |
|---|---|
| **BPE Tokenizer** | Byte-pair encoding trainer + encoder with O(n log n) priority-queue merge and LRU word cache |
| **Transformer LM** | RoPE · RMSNorm · SwiGLU · pre-norm · causal attention — the same building blocks as Llama |
| **Flash Attention** | Full Triton kernel from scratch — forward + two-kernel backward, O(N) memory, 2–4× faster at long context |
| **Optimizer** | AdamW with decoupled weight decay |
| **LR Schedule** | Cosine annealing with linear warmup (LLaMA-style) |
| **Training loop** | Gradient clipping · memmap data loading · W&B logging · checkpointing |
| **Ablations** | Four controlled experiments isolating RoPE, RMSNorm, SwiGLU, and pre-norm |
| **Scaling laws** | Model size sweep · vocab size sweep · token budget sweep · **isoFLOP / Chinchilla curves** |

---

## Scaling law results

I ran the Chinchilla isoFLOP protocol at three compute budgets on OpenWebText.

**IsoFLOP grid** (batch=128, context=512):

| Budget | Small (12.6M) | Medium (25.2M) | Large (42.5M) |
|---|---|---|---|
| C = 1×10¹⁶ | **loss 4.876** ← optimal | 5.176 | 5.305 |
| C = 5×10¹⁶ | **loss 4.092** ← optimal | 4.176 | 4.211 |
| C = 2×10¹⁷ | — | **loss 3.740** ← optimal | 3.742 |

**Power-law fits:**

```
N_opt  =  2796  × C^0.225    (Chinchilla: C^0.50)
D_opt  =  6e-5  × C^0.775    (Chinchilla: C^0.50)
```

At these compute scales, the data strongly favors smaller models trained on more tokens — a much steeper data-optimal regime than Chinchilla's 50/50 split. The exponent gap likely shrinks at larger C; reproducing the crossover is on the roadmap.

---

## Architecture

```
TransformerLM
├── Embedding(vocab_size, d_model)          ← no positional table
├── N × TransformerBlock
│   ├── RMSNorm → CausalMultiHeadSelfAttention (RoPE on Q/K)
│   └── RMSNorm → SwiGLUFFN
└── RMSNorm → Linear(d_model, vocab_size)
```

Default config (~17M params, trains in ~30 min on a free Colab T4):

```python
vocab_size     = 10_000   # from-scratch BPE on TinyStories
context_length = 256
d_model        = 512
num_layers     = 4
num_heads      = 16
d_ff           = 1344     # ceil(8/3 × 512 / 64) × 64  — SwiGLU param-equivalent
```

---

## Ablations & training dynamics

### Learning rate sweep

![LR sweep — val/ppl and val/loss](assets/lr_sweep.png)

`lr=1e-3` (red) beats both `3e-4` and `1e-2` across all steps. At `lr=1e-2` the model still converges but lags significantly — the cosine schedule with warmup is sensitive to the peak LR.

### Batch size sweep

![Batch size sweep — val/ppl](assets/batch_size_sweep.png)

`bs=64` (green) converges fastest in steps but uses more memory; `bs=32` (purple) is the practical sweet spot. `bs=8` (red) takes 19k steps to reach parity — gradient noise from tiny batches requires far more updates.

### Model size sweep

![Model size sweep — val/ppl](assets/model_size_sweep.png)

`size_large` and `size_medium` converge to nearly identical val perplexity (~6.5) given the same token budget, while `size_tiny` plateaus ~10. This is the isoFLOP regime in action: at this compute budget, medium is already capacity-sufficient.

### SwiGLU vs SiLU

![SwiGLU vs SiLU — val/ppl](assets/ablation_swiglu_vs_silu.png)

`normal_swiglu` (purple) consistently outperforms `ablation_silu` (green) by ~0.5 ppl across the full run. SwiGLU's gating mechanism adds ~33% more FFN parameters for the same d_model, which pays off.

### RoPE vs learned positional embeddings

![RoPE ablation — val/ppl](assets/rope_vs_no_rope.png)

`ablation_with_rope` (green) converges to ~20 lower val/ppl than `ablation_no_rope` (purple) by step 4.4k — a large gap that keeps widening. Without RoPE, the model uses a learned absolute position embedding table instead. The attention mechanism loses its sense of relative distance between tokens, so it can't reliably tie subjects to verbs or pronouns to referents across the context window.

The gap in numbers shows up dramatically in generated text — without RoPE the model drifts off-topic within 20 tokens and never recovers.

### Pre-norm vs post-norm

![Pre vs post norm — val/ppl](assets/val_ppl_pre_vs_post.png)

Pre-norm (blue) reaches lower val/ppl faster and holds a consistent ~5–10 ppl lead over post-norm (red) throughout training.

![Pre vs post norm — grad norm](assets/grad_pre_vs_post.png)

The gradient norm plot explains why: post-norm (red) has larger, spikier gradients throughout training — the norm sits *after* the residual, so raw un-normalised activations flow directly into the backward pass. Pre-norm keeps gradients tighter and more consistent (blue), which translates directly into more stable and efficient learning.

### No-norm ablation

![Norm ablation — train/ppl](assets/no_norm_vs_reduced_lr.png)

Without RMSNorm at the default `lr=1e-3`, training explodes (orange, ppl ~9000). Dropping to `lr=1e-4` rescues convergence (blue) but the model learns an order of magnitude slower. Pre-norm RMSNorm is load-bearing.

### Flash Attention vs naive attention

![Flash Attention kernel latency](assets/benchmark_time.png)

Naive attention allocates an N×N score matrix in GPU HBM — memory grows as O(N²). Flash Attention tiles Q, K, V into SRAM blocks and fuses the softmax with the matrix multiply, keeping the working set in fast on-chip memory. The result is O(N) peak memory and a latency crossover that moves from parity to 2–4× faster as sequence length grows past ~1k tokens.

![Flash Attention peak memory](assets/benchmark_memory.png)

Memory savings are dramatic at long context: at seq=4096 the naive kernel uses ~32× more GPU memory than Flash, making context lengths that would OOM under naive attention easily trainable.

![Flash Attention training throughput](assets/benchmark_throughput.png)

End-to-end tokens/sec with a 4-layer, 512-dim model on A100. Flash attention delivers higher throughput at every sequence length ≥ 512, with the gap widening at 2048+ tokens — exactly where batch size would need to shrink under naive attention.

Enable Flash Attention with a single flag:

```bash
python -m transformer_lm.scripts.train \
    --train-tokens ./owt_train_32k.npy \
    --val-tokens   ./owt_val_32k.npy \
    --vocab-size   32000 \
    --context-length 2048 \
    --flash-attention \
    --out-dir ./run-flash
```

Reproduce the benchmark plots (requires a CUDA GPU + `pip install triton`):

```bash
python -m transformer_lm.scripts.benchmark_attention
# writes assets/benchmark_time.png, benchmark_memory.png, benchmark_throughput.png
```

---

**Summary table** (matched compute, OpenWebText 32k):

| Ablation | What changes | Final val ppl |
|---|---|---|
| `baseline` (pre-norm + RoPE) | RoPE · RMSNorm · SwiGLU · pre-norm | ~97 |
| `post_norm` | Norm after residual instead of before | ~105 |
| `no_rope` | Learned absolute position embeddings | ~120 |
| `no_swiglu` | Replace SwiGLU with SiLU FFN | slightly worse |
| `no_rmsnorm` | Remove normalization (lr=1e-4) | diverges at default lr |

---

## Quick start

**Colab (recommended)** — click the badge above. The notebook runs top-to-bottom; each experiment section has a single variable to change (e.g. `ABLATION = "no_rope"`).

**Terminal:**

```bash
git clone https://github.com/kpal002/transformer-lm-from-scratch
cd transformer-lm-from-scratch
pip install -e .
```

**Option A — train from raw text (downloads TinyStories automatically):**

```bash
# Step 1: train BPE tokenizer (~5 min)
python -m transformer_lm.scripts.train_tokenizer --vocab-size 10000 --output-dir ./run

# Step 2: train the model (downloads data, tokenizes, trains — ~30 min on T4)
python -m transformer_lm.scripts.train --out-dir ./run

# Step 3: generate text
python -m transformer_lm.scripts.generate \
    --checkpoint ./run/ckpt_final.pt \
    --vocab ./run/bpe.vocab --merges ./run/bpe.merges \
    --prompt "Once upon a time"
```

**Option B — bring your own pre-tokenized data (no tokenizer files needed):**

```bash
# Pass pre-tokenized .npy files and matching vocab size directly
python -m transformer_lm.scripts.train \
    --train-tokens ./owt_train_32k.npy \
    --val-tokens   ./owt_val_32k.npy \
    --vocab-size   32000 \
    --out-dir      ./run
```

Common overrides:

```bash
# Larger model
python -m transformer_lm.scripts.train \
    --d-model 768 --num-layers 6 --num-heads 16 \
    --num-steps 10000 --out-dir ./run-large

# Resume from checkpoint
python -m transformer_lm.scripts.train --out-dir ./run --resume ./run/ckpt_005000.pt

# Multiple samples
python -m transformer_lm.scripts.generate ... --num-samples 5 --temperature 1.0
```

---

## Project structure

```
transformer_lm/
├── model/
│   ├── attention.py        # softmax, SDPA, RoPE, CausalMultiHeadSelfAttention
│   ├── flash_attention.py  # Flash Attention v2 — Triton forward + backward kernels
│   ├── ffn.py              # SwiGLUFFN
│   ├── layers.py           # Linear, Embedding, RMSNorm
│   └── transformer.py      # TransformerBlock, TransformerLM
├── tokenizer/
│   ├── trainer.py          # BPE training (O(n log n))
│   └── tokenizer.py        # BPETokenizer encode/decode
├── data/
│   └── tinystories.py      # TinyStories downloader
├── training/
│   ├── config.py           # TrainingConfig dataclass
│   ├── loss.py             # cross_entropy_loss (from scratch)
│   ├── optim.py            # AdamW + gradient clipping (from scratch)
│   ├── schedule.py         # cosine LR schedule with warmup
│   ├── data.py             # get_batch, tokenize_and_save
│   ├── checkpoint.py       # save/load checkpoint
│   └── trainer.py          # train(), generate()
└── scripts/
    ├── train_tokenizer.py  # CLI: train BPE tokenizer
    ├── train.py            # CLI: train model end-to-end
    ├── generate.py         # CLI: generate from checkpoint
    └── benchmark_attention.py  # Flash vs naive: latency, memory, throughput
```

---

## Roadmap

- [x] Flash Attention v2 — Triton kernel from scratch (forward + backward)
- [ ] KV cache for fast autoregressive generation
- [ ] SFT + GRPO — reasoning model training on GSM8K

---

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) — Hoffmann et al. (Chinchilla), 2022
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) — Touvron et al., 2023
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — Su et al., 2021
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) — Dao, 2023
- Stanford CS336 — Language Modeling from Scratch
