import torch


def save_checkpoint(model, optimizer, step: int, path: str) -> None:
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step}, path)


def load_checkpoint(path: str, model, optimizer) -> int:
    """Load checkpoint into model and optimizer. Returns the saved step number."""
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["step"]
