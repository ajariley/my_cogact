from typing import Optional, Tuple

import torch


def _validate_path(path: torch.Tensor, name: str) -> None:
    if path.ndim != 4:
        raise ValueError(f"{name} must have shape [N, B, T, C], got {tuple(path.shape)}")
    if path.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one node")
    if not path.is_floating_point():
        raise TypeError(f"{name} must be a floating-point tensor")


def _prepare_action_dim_weights(
    path: torch.Tensor,
    action_dim_weights: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    if action_dim_weights is None:
        return None
    weights = torch.as_tensor(action_dim_weights, device=path.device, dtype=path.dtype)
    if weights.ndim != 1 or weights.shape[0] != path.shape[-1]:
        raise ValueError(
            f"action_dim_weights must have shape [{path.shape[-1]}], got {tuple(weights.shape)}"
        )
    if torch.any(weights < 0) or not torch.any(weights > 0):
        raise ValueError("action_dim_weights must be non-negative and contain a positive value")
    return weights / weights.mean()


def compute_refinement_progress(
    path: torch.Tensor,
    *,
    action_dim_weights: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return per-sample cumulative action refinement progress with shape [N, B]."""
    _validate_path(path, "path")
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")
    if path.shape[0] == 1:
        return path.new_zeros((1, path.shape[1]))

    weights = _prepare_action_dim_weights(path, action_dim_weights)
    squared_delta = (path[1:] - path[:-1]).square()
    if weights is not None:
        squared_delta = squared_delta * weights.view(1, 1, 1, -1)
    step_distance = squared_delta.mean(dim=(-2, -1)).sqrt()  # [N-1, B]
    cumulative = torch.cat(
        [path.new_zeros((1, path.shape[1])), step_distance.cumsum(dim=0)],
        dim=0,
    )
    total = cumulative[-1:]  # [1, B]
    normalized = cumulative / total.clamp_min(eps)

    # A constant path has no data-dependent parameterization. Uniform progress
    # keeps interpolation well-defined while preserving the constant values.
    uniform = torch.linspace(0, 1, path.shape[0], device=path.device, dtype=path.dtype)
    uniform = uniform[:, None].expand_as(normalized)
    return torch.where(total > eps, normalized, uniform)


def resample_path_by_progress(
    path: torch.Tensor,
    progress: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Linearly resample [N, B, T, C] paths at uniform refinement progress."""
    _validate_path(path, "path")
    if progress.shape != path.shape[:2]:
        raise ValueError(
            f"progress must have shape {tuple(path.shape[:2])}, got {tuple(progress.shape)}"
        )
    if num_nodes <= 0:
        raise ValueError(f"num_nodes must be positive, got {num_nodes}")
    if path.shape[0] == 1:
        return path.expand(num_nodes, *path.shape[1:])

    if num_nodes == 1:
        targets = path.new_ones(1)
    else:
        targets = torch.linspace(0, 1, num_nodes, device=path.device, dtype=path.dtype)
    targets = targets[:, None].expand(num_nodes, path.shape[1])  # [M, B]
    progress_by_batch = progress.transpose(0, 1).contiguous()
    targets_by_batch = targets.transpose(0, 1).contiguous()
    upper = torch.searchsorted(progress_by_batch, targets_by_batch, right=False)
    upper = upper.clamp(min=1, max=path.shape[0] - 1)
    lower = upper - 1

    batch_index = torch.arange(path.shape[1], device=path.device)[:, None]
    path_by_batch = path.transpose(0, 1)
    lower_value = path_by_batch[batch_index, lower]
    upper_value = path_by_batch[batch_index, upper]
    lower_progress = progress_by_batch.gather(1, lower)
    upper_progress = progress_by_batch.gather(1, upper)
    fraction = (targets_by_batch - lower_progress) / (upper_progress - lower_progress).clamp_min(
        torch.finfo(path.dtype).eps
    )
    anchors = torch.lerp(lower_value, upper_value, fraction[..., None, None])
    return anchors.transpose(0, 1)


def compress_teacher_path(
    teacher_path: torch.Tensor,
    num_student_nodes: int,
    *,
    action_dim_weights: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compress a teacher path into uniformly spaced refinement anchors."""
    teacher_path = teacher_path.detach()
    progress = compute_refinement_progress(
        teacher_path,
        action_dim_weights=action_dim_weights,
        eps=eps,
    )
    anchors = resample_path_by_progress(teacher_path, progress, num_student_nodes)
    return anchors, progress


def depth_path_losses(
    student_path: torch.Tensor,
    teacher_path: torch.Tensor,
    *,
    action_dim_weights: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
    exclude_terminal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return state loss, macro-update loss, and compressed teacher anchors."""
    _validate_path(student_path, "student_path")
    _validate_path(teacher_path, "teacher_path")
    if student_path.shape[1:] != teacher_path.shape[1:]:
        raise ValueError(
            "teacher/student path payload shapes must match: "
            f"teacher={tuple(teacher_path.shape)}, student={tuple(student_path.shape)}"
        )

    if exclude_terminal:
        num_process_nodes = max(student_path.shape[0] - 1, 0)
        if num_process_nodes == 0 or teacher_path.shape[0] == 1:
            zero = student_path.new_zeros(())
            anchors = teacher_path[-1:].detach().expand(student_path.shape[0], *teacher_path.shape[1:])
            return zero, zero, anchors
        student_process = student_path[:-1]
        teacher_process = teacher_path[:-1]
    else:
        num_process_nodes = student_path.shape[0]
        student_process = student_path
        teacher_process = teacher_path

    teacher_process_anchors, _ = compress_teacher_path(
        teacher_process,
        num_process_nodes,
        action_dim_weights=action_dim_weights,
        eps=eps,
    )
    path_loss = torch.nn.functional.mse_loss(student_process, teacher_process_anchors)
    if student_process.shape[0] == 1:
        macro_loss = student_path.new_zeros(())
    else:
        student_updates = student_process[1:] - student_process[:-1]
        teacher_updates = teacher_process_anchors[1:] - teacher_process_anchors[:-1]
        macro_loss = torch.nn.functional.mse_loss(student_updates, teacher_updates)
    if exclude_terminal:
        teacher_anchors = torch.cat([teacher_process_anchors, teacher_path[-1:].detach()], dim=0)
    else:
        teacher_anchors = teacher_process_anchors
    return path_loss, macro_loss, teacher_anchors
