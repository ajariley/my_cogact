

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, Any

import torch
from torch.utils.data import DataLoader
import draccus

from vla import load_vla, CogACT
from action_model.action_model import ActionModel
from prismatic.vla import get_vla_dataset_and_collator

from conf.distillation import DistillationConfig

def compute_loss(
    teacher_results: Dict[str, Any],
    x0_student_corr: torch.Tensor,
    x0_student_wrong: Optional[torch.Tensor],
    student: ActionModel,
    batch: Dict[str, torch.Tensor],
    cfg: DistillationConfig,
) -> Dict[str, torch.Tensor]:
    """
    总损失 = λ_task * L_task + λ_distill * L_distill
    L_task = action_model.loss(actions, z)  # 扩散自身，真实数据
    L_distill = λ_final*L_final + λ_traj*L_traj + [λ_neg*L_neg]
    L_neg：指令敏感，当 x0_student_wrong 非空时，margin/contrast 使 student(corr) 与 student(wrong) 输出距离足够大
    返回：{"total": ..., "task": ..., "distill": ..., "final": ..., "traj": ..., "neg": ...}
    """
    # TODO: 实现 只做老师和学生的mse
    x0_teacher = teacher_results["x0_teacher"]
    x0_student = x0_student_corr
    loss_mse = torch.nn.functional.mse_loss(x0_teacher, x0_student)
    return {"total": loss_mse, "task": loss_mse, "distill": loss_mse}


    