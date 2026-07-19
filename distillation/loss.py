

from typing import Dict

import torch

from action_model.action_model import ActionModel

from conf.distillation import DistillationConfig

def compute_loss(
    student: ActionModel,
    actions_future: torch.Tensor,
    z_corr: torch.Tensor,
    x0_teacher: torch.Tensor,
    x0_student: torch.Tensor,
    teacher_trajectory: torch.Tensor,
    student_trajectory: torch.Tensor,
    cfg: DistillationConfig,
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
    loss_total = cfg.lambda_task * loss_task + cfg.lambda_final * loss_final + cfg.lambda_traj * loss_traj
    return {
        "total": loss_total,
        "task": loss_task,
        "final": loss_final,
        "traj": loss_traj,
    }


    
