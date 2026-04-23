

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, Any

import torch
from torch.utils.data import DataLoader
import draccus

from vla import load_vla, CogACT
from action_model.action_model import ActionModel
from prismatic.vla import get_vla_dataset_and_collator


@dataclass
class DistillationConfig:
    teacher_checkpoint: Path = Path("/home/huangjiaqi/projects/CogACT-Base/checkpoints/CogACT-Base.pt")
    data_root_dir: Path = Path("datasets/open-x-embodiment")
    data_mix: str = "bridge"
    output_dir: Path = Path("runs/distillation")
    batch_size: int = 4
    max_batches: Optional[int] = None
    epochs: int = 10
    lr: float = 1e-4
    num_ddim_steps_teacher: int = 10
    num_ddim_steps_student: int = 4
    action_model_type_teacher: str = "DiT-B"
    action_model_type_student: str = "DiT-S"
    future_action_window_size: int = 15
    past_action_window_size: int = 0
    action_dim: int = 7
    cfg_scale_teacher: float = 1.5  # 仅教师推理时用 CFG，学生不用
    # FSDP 相关参数
    use_fsdp: bool = True
    use_cpu_offload: bool = False  # True 时部分权重留 CPU，易与手写 .to(cuda) 的 batch 设备不一致
    use_bf16: bool = True
    reduce_in_full_precision: bool = True
    train_strategy: str = "fsdp"
    # Loss 权重
    lambda_task: float = 0.3
    lambda_final: float = 1.0
    lambda_traj: float = 0.5
    lambda_neg: float = 0.1
    use_instruction_constraint: bool = True  # 指令敏感：同图+错误指令→z_wrong，L_neg 拉大学生(corr)与(wrong)距离
    log_memory: bool = False
    log_memory_tf: bool = True

    # 在 DistillationConfig 中添加（参考 conf/vla.py 的 shuffle_buffer_size）
    shuffle_buffer_size: int = 256_000   # bridge 用 256k，oxe_magic_soup 用 250k
    image_aug: bool = False
    load_all_data_for_training: bool = True

