

from typing import Dict, Optional

import torch

from action_model.action_model import ActionModel

from conf.distillation import DistillationConfig
from distillation.path import depth_path_losses

def compute_loss(
    student: ActionModel,
    actions_future: torch.Tensor,
    z_corr: torch.Tensor,
    x0_teacher: torch.Tensor,
    x0_student: torch.Tensor,
    teacher_trajectory: torch.Tensor,
    student_trajectory: torch.Tensor,
    cfg: DistillationConfig,
    teacher_depth_path: Optional[torch.Tensor] = None,
    student_depth_path: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute the current distillation training loss.

    L_task trains the student diffusion head on real future actions.
    L_final aligns the student's final DDIM sample with the teacher's final sample.
    """
    if x0_teacher.shape != x0_student.shape or x0_student.shape != actions_future.shape:
        raise RuntimeError(
            "loss shape mismatch: "
            f"actions_future={tuple(actions_future.shape)}, "
            f"x0_teacher={tuple(x0_teacher.shape)}, "
            f"x0_student={tuple(x0_student.shape)}"
        )
    if teacher_trajectory.shape != student_trajectory.shape:
        raise RuntimeError(
            "trajectory shape mismatch: "
            f"teacher_trajectory={tuple(teacher_trajectory.shape)}, "
            f"student_trajectory={tuple(student_trajectory.shape)}"
        )

    loss_task = student.loss(actions_future, z_corr)
    loss_final = torch.nn.functional.mse_loss(x0_student, x0_teacher.detach())
    loss_traj = torch.nn.functional.mse_loss(
        student_trajectory,
        teacher_trajectory.detach(),
    )
    loss_path = x0_student.new_zeros(())
    loss_macro = x0_student.new_zeros(())
    if teacher_depth_path is not None and student_depth_path is not None:
        action_dim_weights = getattr(cfg, "refinement_action_dim_weights", None)
        loss_path, loss_macro, _ = depth_path_losses(
            student_depth_path,
            teacher_depth_path,
            action_dim_weights=action_dim_weights,
            eps=cfg.refinement_progress_eps,
        )
    elif cfg.lambda_path != 0 or cfg.lambda_macro != 0:
        raise RuntimeError(
            "teacher_depth_path and student_depth_path are required when depth path losses are enabled"
        )

    loss_total = (
        cfg.lambda_task * loss_task
        + cfg.lambda_final * loss_final
        + cfg.lambda_traj * loss_traj
        + cfg.lambda_path * loss_path
        + cfg.lambda_macro * loss_macro
    )
    return {
        "total": loss_total,
        "task": loss_task,
        "final": loss_final,
        "traj": loss_traj,
        "path": loss_path,
        "macro": loss_macro,
    }


    
