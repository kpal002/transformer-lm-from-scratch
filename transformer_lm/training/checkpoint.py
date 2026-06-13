import torch


def save_checkpoint(model, optimizer, step: int, path: str, meta: dict | None = None) -> None:
    payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step}
    if meta:
        payload["meta"] = meta
    torch.save(payload, path)


def load_checkpoint(path: str, model, optimizer) -> int:
    """Load checkpoint into model and optimizer. Returns the saved step number."""
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["step"]
