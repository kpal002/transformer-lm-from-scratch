"""Core training loop and autoregressive text generation.

train()    — runs the full training loop with logging, validation, checkpointing,
             and optional W&B integration.
generate() — autoregressive sampling from a trained model with temperature
             scaling and nucleus (top-p) sampling.
"""

from __future__ import annotations

import json
import math
import os
import time

import numpy as np
import torch

from .checkpoint import load_checkpoint, save_checkpoint
from .config import TrainingConfig
from .data import get_batch
from .loss import cross_entropy_loss
from .optim import AdamW, clip_gradient_norm
from .schedule import get_lr_cosine_schedule


@torch.no_grad()
def estimate_val_loss(model, val_data: np.ndarray, cfg: TrainingConfig, device: str) -> float:
    """Estimate validation loss by averaging over multiple random batches.

    Using multiple batches rather than a single large batch gives a lower-
    variance estimate without requiring the entire validation set to fit in
    GPU memory.

    Wrapped in @torch.no_grad() to skip gradient computation and reduce
    memory usage during evaluation.

    Args:
        model:    Trained TransformerLM in any state (set to eval internally).
        val_data: Tokenized validation set as a 1-D numpy array.
        cfg:      Training config (reads batch_size, context_length, val_steps).
        device:   PyTorch device string.

    Returns:
        Mean cross-entropy loss across cfg.val_steps random batches.
    """
    model.eval()
    total = 0.0
    for _ in range(cfg.val_steps):
        inputs, targets = get_batch(val_data, cfg.batch_size, cfg.context_length, device)
        total += cross_entropy_loss(model(inputs), targets).item()
    model.train()
    return total / cfg.val_steps


def _meta(cfg: TrainingConfig) -> dict:
    """Extract model shape metadata to store in checkpoints.

    generate.py uses this to reconstruct the model architecture from a
    checkpoint without requiring the user to re-specify hyperparameters.
    """
    return {
        "context_length":  cfg.context_length,
        "num_heads":       cfg.num_heads,
        "norm_type":       cfg.norm_type,
        "use_rope":        cfg.use_rope,
        "use_flash_attn":  cfg.use_flash_attn,
    }


def train(
    model,
    optimizer: AdamW,
    train_data: np.ndarray,
    val_data: np.ndarray,
    cfg: TrainingConfig,
    device: str,
) -> None:
    """Run the full training loop.

    Each step:
      1. Sample a random batch of (input, target) pairs.
      2. Forward pass → logits → cross-entropy loss.
      3. Backward pass → gradients.
      4. Clip gradients by global L2 norm.
      5. Update learning rate via cosine schedule.
      6. AdamW optimizer step.
      7. Zero gradients.
      8. Log metrics (periodically to stdout and W&B).
      9. Validate (periodically).
     10. Save checkpoint (periodically + always at the end).

    Args:
        model:      TransformerLM instance, already moved to `device`.
        optimizer:  AdamW optimizer instance.
        train_data: Tokenized training set as a 1-D numpy array.
        val_data:   Tokenized validation set as a 1-D numpy array.
        cfg:        All hyperparameters (see TrainingConfig for field docs).
        device:     PyTorch device string ("cpu", "cuda", "mps").
    """
    os.makedirs(cfg.out_dir, exist_ok=True)
    model.train()

    metrics_path = os.path.join(cfg.out_dir, "metrics.jsonl")
    metrics_file = open(metrics_path, "a")

    # Initialise W&B run if requested
    run = None
    if cfg.use_wandb:
        import wandb
        run = wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run_name,
            config=vars(cfg),   # log all hyperparameters
            resume="allow",     # allows resuming a previous run by name
        )
        print(f"W&B run: {run.url}")

    # Resume from checkpoint if specified
    start_step = 0
    if cfg.resume is not None:
        start_step = load_checkpoint(cfg.resume, model, optimizer)
        print(f"Resumed from step {start_step}")

    t0 = time.time()

    for step in range(start_step, cfg.num_steps):
        # ── Forward pass ──────────────────────────────────────────────────────
        inputs, targets = get_batch(train_data, cfg.batch_size, cfg.context_length, device)
        logits = model(inputs)
        loss = cross_entropy_loss(logits, targets)
        loss.backward()

        # ── Gradient clipping ─────────────────────────────────────────────────
        # Clip before the optimizer step so the update is bounded.
        # Returns the pre-clip norm for logging.
        grad_norm = clip_gradient_norm(model.parameters(), cfg.max_grad_norm)

        # ── Learning rate update ──────────────────────────────────────────────
        # Set lr on all param groups before the optimizer step, so the step
        # uses the schedule value for this particular step index.
        lr = get_lr_cosine_schedule(
            t=step,
            alpha_max=cfg.alpha_max,
            alpha_min=cfg.alpha_min,
            T_w=cfg.warmup_steps,
            T_c=cfg.num_steps,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        # ── Optimizer step ────────────────────────────────────────────────────
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)  # set_to_none saves a memset vs fill_zero

        elapsed = time.time() - t0

        # ── Logging ───────────────────────────────────────────────────────────
        if step % cfg.log_every == 0:
            train_loss = loss.item()
            print(
                f"step {step:5d} | loss {train_loss:.4f} | ppl {math.exp(train_loss):7.2f} "
                f"| lr {lr:.2e} | grad {grad_norm:.3f} | {elapsed/60:.1f} min"
            )
            record: dict = {
                "step": step, "train_loss": train_loss,
                "train_ppl": math.exp(train_loss), "lr": lr,
                "grad_norm": grad_norm, "wall_clock_sec": elapsed,
            }
            if run is not None:
                import wandb
                wandb.log(
                    {
                        "train/loss":      train_loss,
                        "train/ppl":       math.exp(train_loss),
                        "train/lr":        lr,
                        "train/grad_norm": grad_norm,
                        "wall_clock_sec":  elapsed,
                    },
                    step=step,
                )

        # ── Validation ────────────────────────────────────────────────────────
        if step % cfg.val_every == 0 and step > 0:
            vl = estimate_val_loss(model, val_data, cfg, device)
            print(f"  [val] loss {vl:.4f} | ppl {math.exp(vl):.2f}")
            record["val_loss"] = vl
            record["val_ppl"]  = math.exp(vl)
            if run is not None:
                import wandb
                wandb.log(
                    {"val/loss": vl, "val/ppl": math.exp(vl), "wall_clock_sec": elapsed},
                    step=step,
                )

        if step % cfg.log_every == 0:
            metrics_file.write(json.dumps(record) + "\n")
            metrics_file.flush()

        # ── Periodic checkpoint ───────────────────────────────────────────────
        if step % cfg.save_every == 0 and step > 0:
            path = os.path.join(cfg.out_dir, f"ckpt_{step:06d}.pt")
            save_checkpoint(model, optimizer, step, path, _meta(cfg))
            print(f"  [ckpt] saved → {path}")

    # Always save a final checkpoint regardless of save_every alignment
    save_checkpoint(
        model, optimizer, cfg.num_steps,
        os.path.join(cfg.out_dir, "ckpt_final.pt"),
        _meta(cfg),
    )
    metrics_file.close()
    print("Training complete.")

    if run is not None:
        import wandb
        wandb.finish()


@torch.no_grad()
def generate(
    model,
    prompt_tokens: list[int],
    context_length: int,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_p: float = 0.95,
    eos_token_id: int | None = None,
    device: str = "cpu",
) -> list[int]:
    """Autoregressively generate tokens from a trained model.

    Sampling strategy: temperature scaling + nucleus (top-p) sampling.

    Temperature scaling:
        Divides logits by `temperature` before softmax.
        - temperature < 1.0 → sharper distribution → more predictable output
        - temperature > 1.0 → flatter distribution → more random/creative output
        - temperature = 1.0 → unmodified model distribution

    Nucleus (top-p) sampling (Holtzman et al., 2020):
        Instead of sampling from the full vocabulary, restrict to the smallest
        set of tokens whose cumulative probability exceeds `top_p`.  This avoids
        sampling very unlikely tokens (which produce incoherent text) while
        preserving diversity in the high-probability head of the distribution.

    Context window management:
        If the sequence grows beyond context_length, only the last
        context_length tokens are fed to the model — older context is
        discarded.  (A KV cache would avoid re-computing these tokens.)

    Args:
        model:          Trained TransformerLM (eval mode set internally).
        prompt_tokens:  List of integer token IDs representing the prompt.
        context_length: Model's maximum context window.
        max_new_tokens: Maximum number of tokens to generate.
        temperature:    Logit temperature (default 0.8).
        top_p:          Nucleus sampling threshold (default 0.95).
        eos_token_id:   If provided, generation stops when this token is sampled.
        device:         PyTorch device string.

    Returns:
        List of generated token IDs (prompt tokens are NOT included).
    """
    model.eval()
    tokens = list(prompt_tokens)
    generated: list[int] = []

    for _ in range(max_new_tokens):
        # Truncate to the model's context window
        context = tokens[-context_length:]
        input_ids = torch.tensor([context], dtype=torch.long, device=device)

        # Forward pass: get logits for the last position only
        logits = model(input_ids)
        next_tok_logits = logits[0, -1] / temperature   # (vocab_size,)

        # Convert to probabilities
        probs = torch.softmax(next_tok_logits, dim=-1)

        # Nucleus sampling: sort descending, find cumulative cutoff
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)

        # Zero out tokens whose cumulative probability exceeds top_p
        # (shift by one so the token that crosses the threshold is kept)
        sorted_probs[cumulative - sorted_probs > top_p] = 0.0
        sorted_probs = sorted_probs / sorted_probs.sum()  # renormalise

        # Sample one token from the nucleus
        next_token = sorted_idx[torch.multinomial(sorted_probs, num_samples=1)].item()

        tokens.append(next_token)
        generated.append(next_token)

        # Stop if end-of-sequence token is generated
        if eos_token_id is not None and next_token == eos_token_id:
            break

    return generated
