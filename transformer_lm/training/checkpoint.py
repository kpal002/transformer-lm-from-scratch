"""Checkpoint save and load utilities.

A checkpoint contains three things:
  - model state dict  — all parameter tensors
  - optimizer state   — Adam moment buffers (m, v) and step counters per param
  - step              — the training step at which it was saved
  - meta (optional)   — architecture hyperparameters needed to reconstruct the
                        model without re-specifying CLI flags

Saving optimizer state is essential for resuming training correctly.  Without
it, the Adam moment buffers restart from zero, which effectively resets the
effective learning rate for each parameter and can cause a loss spike.
"""

import torch


def save_checkpoint(
    model,
    optimizer,
    step: int,
    path: str,
    meta: dict | None = None,
) -> None:
    """Serialize model weights, optimizer state, and training step to disk.

    Args:
        model:     TransformerLM — its state_dict is saved.
        optimizer: AdamW optimizer — its state_dict (moments + steps) is saved.
        step:      Current training step, used to resume at the right iteration.
        path:      File path to write the .pt checkpoint.
        meta:      Optional dict of architecture hyperparameters
                   (context_length, num_heads, norm_type, use_rope).
                   Stored so generate.py can reconstruct the model without
                   requiring the user to re-pass all CLI flags.
    """
    payload = {
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step":      step,
    }
    if meta:
        payload["meta"] = meta
    torch.save(payload, path)


def load_checkpoint(path: str, model, optimizer) -> int:
    """Load a checkpoint into an already-instantiated model and optimizer.

    The model and optimizer must already have the correct architecture and
    parameter count — this function only fills in the weights and moments,
    it does not reconstruct the objects.

    Args:
        path:      Path to the .pt checkpoint file.
        model:     TransformerLM instance — weights are loaded in-place.
        optimizer: AdamW instance — moment buffers are loaded in-place.

    Returns:
        The training step stored in the checkpoint, so training can resume
        from the next step.
    """
    ckpt = torch.load(path, map_location="cpu")   # load to CPU first, then .to(device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["step"]
