

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, Any

import torch
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import draccus

from vla import load_vla, CogACT
from action_model.action_model import ActionModel
from prismatic.vla import get_vla_dataset_and_collator


class ShardedIterableDataset(IterableDataset):
    def __init__(self, dataset: IterableDataset, rank: int, world_size: int) -> None:
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        for idx, sample in enumerate(self.dataset):
            if idx % self.world_size == self.rank:
                yield sample

    def __len__(self) -> int:
        if hasattr(self.dataset, "__len__"):
            return len(self.dataset) // self.world_size
        raise TypeError("ShardedIterableDataset length is unknown because the wrapped dataset has no __len__")


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
    sampler = None
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        if isinstance(vla_dataset, IterableDataset):
            vla_dataset = ShardedIterableDataset(
                vla_dataset,
                rank=dist.get_rank(),
                world_size=dist.get_world_size(),
            )
            print(
                f"[distillation] IterableDataset sharded by rank: "
                f"rank={dist.get_rank()} world_size={dist.get_world_size()}",
                flush=True,
            )
        elif hasattr(vla_dataset, "__len__"):
            sampler = DistributedSampler(
                vla_dataset,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=True,
                drop_last=False,
            )
        else:
            print(
                "[distillation] WARNING: dataset does not expose __len__; DistributedSampler cannot be used. "
                "Ensure the dataset pipeline shards by rank, otherwise ranks may see duplicate data.",
                flush=True,
            )

    dataloader = DataLoader(
        vla_dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collator,
        num_workers=0,
    )
    return dataloader
