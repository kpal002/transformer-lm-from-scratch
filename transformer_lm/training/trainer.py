from __future__ import annotations

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
    model.eval()
    total = 0.0
    for _ in range(cfg.val_steps):
        inputs, targets = get_batch(val_data, cfg.batch_size, cfg.context_length, device)
        total += cross_entropy_loss(model(inputs), targets).item()
    model.train()
    return total / cfg.val_steps


def train(
    model,
    optimizer: AdamW,
    train_data: np.ndarray,
    val_data: np.ndarray,
    cfg: TrainingConfig,
    device: str,
) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)
    model.train()

    run = None
    if cfg.use_wandb:
        import wandb
        run = wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run_name,
            config=vars(cfg),
            resume="allow",
        )
        print(f"W&B run: {run.url}")

    start_step = 0
    if cfg.resume is not None:
        start_step = load_checkpoint(cfg.resume, model, optimizer)
        print(f"Resumed from step {start_step}")

    t0 = time.time()

    for step in range(start_step, cfg.num_steps):
        inputs, targets = get_batch(train_data, cfg.batch_size, cfg.context_length, device)
        logits = model(inputs)
        loss = cross_entropy_loss(logits, targets)
        loss.backward()

        grad_norm = clip_gradient_norm(model.parameters(), cfg.max_grad_norm)

        lr = get_lr_cosine_schedule(
            t=step, alpha_max=cfg.alpha_max, alpha_min=cfg.alpha_min,
            T_w=cfg.warmup_steps, T_c=cfg.num_steps,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        elapsed = time.time() - t0

        if step % cfg.log_every == 0:
            train_loss = loss.item()
            print(
                f"step {step:5d} | loss {train_loss:.4f} | ppl {math.exp(train_loss):7.2f} "
                f"| lr {lr:.2e} | grad {grad_norm:.3f} | {elapsed/60:.1f} min"
            )
            if run is not None:
                import wandb
                wandb.log(
                    {"train/loss": train_loss, "train/ppl": math.exp(train_loss),
                     "train/lr": lr, "train/grad_norm": grad_norm, "wall_clock_sec": elapsed},
                    step=step,
                )

        if step % cfg.val_every == 0 and step > 0:
            vl = estimate_val_loss(model, val_data, cfg, device)
            print(f"  [val] loss {vl:.4f} | ppl {math.exp(vl):.2f}")
            if run is not None:
                import wandb
                wandb.log({"val/loss": vl, "val/ppl": math.exp(vl), "wall_clock_sec": elapsed}, step=step)

        if step % cfg.save_every == 0 and step > 0:
            path = os.path.join(cfg.out_dir, f"ckpt_{step:06d}.pt")
            save_checkpoint(model, optimizer, step, path)
            print(f"  [ckpt] saved → {path}")

    save_checkpoint(model, optimizer, cfg.num_steps, os.path.join(cfg.out_dir, "ckpt_final.pt"))
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
    """Autoregressive generation with temperature scaling and top-p nucleus sampling.

    Args:
        model:          Trained TransformerLM (eval mode set internally).
        prompt_tokens:  List of integer token IDs.
        context_length: Model's maximum context window.
        max_new_tokens: Maximum tokens to generate.
        temperature:    Logit scaling — lower = sharper, higher = more random.
        top_p:          Nucleus sampling threshold.
        eos_token_id:   Stop early when this token is sampled.
        device:         PyTorch device string.

    Returns:
        List of generated token IDs (prompt not included).
    """
    model.eval()
    tokens = list(prompt_tokens)
    generated = []

    for _ in range(max_new_tokens):
        context = tokens[-context_length:]
        input_ids = torch.tensor([context], dtype=torch.long, device=device)

        logits = model(input_ids)
        next_tok_logits = logits[0, -1] / temperature
        probs = torch.softmax(next_tok_logits, dim=-1)

        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        sorted_probs[cumulative - sorted_probs > top_p] = 0.0
        sorted_probs = sorted_probs / sorted_probs.sum()

        next_token = sorted_idx[torch.multinomial(sorted_probs, num_samples=1)].item()
        tokens.append(next_token)
        generated.append(next_token)

        if eos_token_id is not None and next_token == eos_token_id:
            break

    return generated
