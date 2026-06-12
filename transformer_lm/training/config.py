from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainingConfig:
    # ── Model ──────────────────────────────────────────────────────────────────
    vocab_size:      int   = 10_000
    context_length:  int   = 256
    d_model:         int   = 512
    num_layers:      int   = 4
    num_heads:       int   = 16
    d_ff:            int   = 1344    # ceil(8/3 × 512 / 64) × 64
    theta:           float = 10_000.0

    # ── Ablation flags ─────────────────────────────────────────────────────────
    norm_type:       str   = "pre"    # "pre" | "post" | "none"
    use_rope:        bool  = True
    ffn_type:        str   = "swiglu" # "swiglu" | "silu"

    # ── Optimizer ──────────────────────────────────────────────────────────────
    alpha_max:       float = 1e-3
    alpha_min:       float = 1e-4
    beta1:           float = 0.9
    beta2:           float = 0.999
    eps:             float = 1e-8
    weight_decay:    float = 0.1
    max_grad_norm:   float = 1.0

    # ── Schedule ───────────────────────────────────────────────────────────────
    num_steps:       int   = 5_000
    warmup_steps:    int   = 200

    # ── Data ───────────────────────────────────────────────────────────────────
    batch_size:      int   = 32

    # ── Logging & checkpointing ────────────────────────────────────────────────
    log_every:       int   = 50
    val_every:       int   = 500
    val_steps:       int   = 20
    save_every:      int   = 1_000
    out_dir:         str   = "checkpoints"
    resume:          Optional[str] = None

    # ── Weights & Biases ───────────────────────────────────────────────────────
    use_wandb:       bool  = False
    wandb_project:   str   = "transformer-lm"
    wandb_run_name:  Optional[str] = None
