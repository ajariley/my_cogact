

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, Any

import torch
from torch.utils.data import DataLoader
import draccus

from vla import load_vla, CogACT
from action_model.action_model import ActionModel
from prismatic.vla import get_vla_dataset_and_collator


def load_teacher(
    checkpoint_path: Path,
    action_model_type: str,
    future_action_window_size: int,
    hf_token: Optional[str] = None,
    **kwargs,
) -> CogACT:
    """
    加载教师 CogACT，eval + 冻结。
    调用：load_vla(...)
    """
    teacher = load_vla(
        str(checkpoint_path),
        hf_token=hf_token,
        load_for_training=True,  # 与原版一致，走 HF 分片加载路径，降低 FSDP init 时 OOM 风险
        action_model_type=action_model_type,
        future_action_window_size=future_action_window_size,
        **kwargs,
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


def load_student(
    token_size: int,
    action_model_type: str,
    in_channels: int,
    future_action_window_size: int,
    past_action_window_size: int,
    device: torch.device,
) -> ActionModel:
    """
    创建学生 ActionModel（如 DiT-S）。
    调用：ActionModel(token_size=..., model_type=..., ...)
    """
    student = ActionModel(
        token_size=token_size,
        model_type=action_model_type,
        in_channels=in_channels,
        future_action_window_size=future_action_window_size,
        past_action_window_size=past_action_window_size,
    )
    student.create_ddim(ddim_step=4)  # 示例：4 步
    student = student.to(device)
    return student


def load_dataloader(
    teacher,
    data_root_dir: Path,
    data_mix: str,
    batch_size: int,
    **kwargs,
) -> DataLoader:
    """
    加载 dataset + collator，创建 DataLoader。
    调用：get_vla_dataset_and_collator(..., image_transform=teacher.vision_backbone.get_image_transform(), ...)
    """
    vla_dataset, _, collator = get_vla_dataset_and_collator(
        data_root_dir,
        data_mix,
        image_transform=teacher.vision_backbone.get_image_transform(),
        tokenizer=teacher.llm_backbone.get_tokenizer(),
        prompt_builder_fn=teacher.llm_backbone.prompt_builder_fn,
        default_image_resolution=teacher.vision_backbone.default_image_resolution,
        shuffle_buffer_size=kwargs.get("shuffle_buffer_size", 256_000),
        image_aug=kwargs.get("image_aug", False),
        load_all_data_for_training=kwargs.get("load_all_data_for_training", True),
        future_action_window_size=kwargs.get("future_action_window_size", 15),
        past_action_window_size=kwargs.get("past_action_window_size", 0),
        **{k: v for k, v in kwargs.items() if k not in (
            "future_action_window_size", "past_action_window_size",
            "shuffle_buffer_size", "image_aug", "load_all_data_for_training",
        )},
    )
    dataloader = DataLoader(
        vla_dataset,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=0,
    )
    return dataloader
