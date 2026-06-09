from pathlib import Path
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional

import torch


def student_state_dict(student: torch.nn.Module) -> Dict[str, Any]:
    state = student.state_dict()
    normalized = {}
    for key, value in state.items():
        if key.startswith("net.module."):
            normalized["net." + key[len("net.module."):]] = value
        elif key.startswith("module."):
            normalized[key[len("module."):]] = value
        else:
            normalized[key] = value
    return normalized


def _state_dict_for_student(student: torch.nn.Module, state: Dict[str, Any]) -> Dict[str, Any]:
    student_keys = set(student.state_dict().keys())
    if any(key.startswith("net.module.") for key in student_keys):
        return {
            "net.module." + key[len("net."):] if key.startswith("net.") else key: value
            for key, value in state.items()
        }
    if any(key.startswith("module.") for key in student_keys):
        return {
            "module." + key if not key.startswith("module.") else key: value
            for key, value in state.items()
        }
    return {
        "net." + key[len("net.module."):] if key.startswith("net.module.") else
        key[len("module."):] if key.startswith("module.") else key: value
        for key, value in state.items()
    }


def _to_checkpoint_config(cfg: Any) -> Dict[str, Any]:
    if is_dataclass(cfg):
        raw = asdict(cfg)
    elif isinstance(cfg, dict):
        raw = dict(cfg)
    else:
        raw = dict(getattr(cfg, "__dict__", {}))
    return {k: str(v) if isinstance(v, Path) else v for k, v in raw.items()}


def save_checkpoint(
    path: Path,
    student: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    cfg: Any,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "student": student_state_dict(student),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "cfg": _to_checkpoint_config(cfg),
        },
        path,
    )


def load_checkpoint(
    path: Optional[Path],
    student: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    if path is None:
        return {"epoch": 0, "step": 0, "loaded": False}

    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    student.load_state_dict(_state_dict_for_student(student, checkpoint["student"]))
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return {
        "epoch": int(checkpoint.get("epoch", 0)),
        "step": int(checkpoint.get("step", 0)),
        "loaded": True,
        "path": path,
    }
