"""TrainingConfig: central configuration dataclass for all training runs.

Keeping every hyperparameter in one place makes it easy to log to W&B,
save alongside checkpoints, and reproduce experiments.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """All hyperparameters for one training run.

    Fields are grouped by concern.  Defaults reproduce the baseline TinyStories
    experiment: ~17M-parameter model, 5k steps, ~30 min on a free Colab T4.
    """

    # ── Model architecture ────────────────────────────────────────────────────
    vocab_size: int = 10_000
    """Number of tokens in the BPE vocabulary.  Must match the tokenizer used
    to produce the training data."""

    context_length: int = 256
    """Maximum sequence length (tokens).  Determines the size of the causal
    mask and the RoPE pre-computed tables.  Also sets the positional embedding
    table size when use_rope=False."""

    d_model: int = 512
    """Residual stream width.  All attention and FFN sub-layers operate in this
    dimension.  Larger values → more expressive but more memory and compute."""

    num_layers: int = 4
    """Number of stacked TransformerBlocks.  Depth scales model capacity and
    compute together."""

    num_heads: int = 16
    """Attention heads per block.  d_model must be divisible by num_heads.
    Per-head dimension d_k = d_model / num_heads = 32 at default settings."""

    d_ff: int = 1344
    """SwiGLU inner dimension.  Default = ceil(8/3 × 512 / 64) × 64, which
    keeps parameter count parameter-equivalent to a standard 4× FFN."""

    theta: float = 10_000.0
    """RoPE base frequency.  Controls how fast position encodings rotate.
    Larger values extend the effective context range (LLaMA 3 uses 500,000)."""

    # ── Ablation flags ────────────────────────────────────────────────────────
    norm_type: str = "pre"
    """Normalisation placement: "pre" (LLaMA-style, most stable),
    "post" (original Transformer), or "none" (no normalisation — ablation only,
    typically diverges at default lr)."""

    use_rope: bool = True
    """Whether to use Rotary Positional Embedding.  When False, a learned
    positional embedding table is added at the input layer instead."""

    ffn_type: str = "swiglu"
    """FFN variant: "swiglu" (default, LLaMA-style) or "silu" (ablation)."""

    # ── Optimizer (AdamW) ─────────────────────────────────────────────────────
    alpha_max: float = 1e-3
    """Peak learning rate, reached at the end of the warmup phase."""

    alpha_min: float = 1e-4
    """Minimum learning rate, held constant after the cosine decay completes."""

    beta1: float = 0.9
    """AdamW first-moment decay (exponential moving average of gradients)."""

    beta2: float = 0.999
    """AdamW second-moment decay (EMA of squared gradients).  Higher than the
    Adam paper default (0.999 vs 0.999) to reduce noise in variance estimates."""

    eps: float = 1e-8
    """AdamW epsilon: small constant in the denominator to prevent division by
    zero when v̂ is very small."""

    weight_decay: float = 0.1
    """Decoupled weight decay coefficient.  0.1 is the LLaMA / GPT-3 standard —
    aggressive but beneficial for generalisation."""

    max_grad_norm: float = 1.0
    """Global gradient clipping threshold.  Gradients with L2 norm larger than
    this are rescaled.  1.0 is standard for LLM training."""

    # ── LR schedule ───────────────────────────────────────────────────────────
    num_steps: int = 5_000
    """Total training steps (= total gradient updates).  One step = one
    forward + backward + optimizer update on one batch."""

    warmup_steps: int = 200
    """Number of linear warmup steps at the beginning of training.  lr grows
    from 0 to alpha_max linearly over this many steps."""

    # ── Data ──────────────────────────────────────────────────────────────────
    batch_size: int = 32
    """Number of sequences per batch.  Total tokens per step =
    batch_size × context_length = 32 × 256 = 8,192 at default settings."""

    # ── Logging & checkpointing ───────────────────────────────────────────────
    log_every: int = 50
    """Print training loss and stats every N steps."""

    val_every: int = 500
    """Run validation and log val/loss every N steps."""

    val_steps: int = 20
    """Number of random batches to average for the validation loss estimate.
    Larger values give a more accurate estimate but cost more time."""

    save_every: int = 1_000
    """Save a checkpoint every N steps.  The final checkpoint is always saved."""

    out_dir: str = "checkpoints"
    """Directory where checkpoints are written."""

    resume: Optional[str] = None
    """Path to a checkpoint to resume from.  If None, training starts fresh."""

    # ── Weights & Biases ──────────────────────────────────────────────────────
    use_wandb: bool = False
    """Enable W&B logging.  Requires `wandb login` before training."""

    wandb_project: str = "transformer-lm"
    """W&B project name.  All runs under the same project are grouped together
    in the W&B dashboard."""

    wandb_run_name: Optional[str] = None
    """W&B run name.  If None, W&B generates a random name.  Use descriptive
    names like "ablation_no_rope" to make the dashboard readable."""
