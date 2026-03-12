
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
from distillation.loaders import load_teacher, load_student, load_dataloader
from distillation.runners import run_teacher_with_recording, run_student_ddim
from distillation.loss import compute_loss

"""是主训练函数，用于训练学生模型。入口函数调用这个函数"""

def train_distillation(cfg: DistillationConfig, hf_token: Optional[str] = None) -> None:
    """
    主训练入口。
    流程：load_teacher → load_student → load_dataloader → for batch in dataloader { ... }
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # A. 加载
    teacher = load_teacher(
        cfg.teacher_checkpoint,
        cfg.action_model_type_teacher,
        cfg.future_action_window_size,
        hf_token=hf_token,
    )
    teacher = teacher.to(device)

    token_size = teacher.llm_backbone.llm.lm_head.in_features  # 4096
    student = load_student(
        token_size=token_size,
        action_model_type=cfg.action_model_type_student,
        in_channels=cfg.action_dim,
        future_action_window_size=cfg.future_action_window_size,
        past_action_window_size=cfg.past_action_window_size,
        device=device,
    )

    dataloader = load_dataloader(
        teacher,
        cfg.data_root_dir,
        cfg.data_mix,
        cfg.batch_size,
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
        load_all_data_for_training=cfg.load_all_data_for_training,
        future_action_window_size=cfg.future_action_window_size,
        past_action_window_size=cfg.past_action_window_size,
    )

    optimizer = torch.optim.AdamW(student.parameters(), lr=cfg.lr)

    # B. 训练
    batch_count = 0
    for epoch in range(cfg.epochs):
        for batch in dataloader:
            if cfg.max_batches is not None and batch_count >= cfg.max_batches:
                return#提前停下，以供debug
            batch_count += 1

            # 将 batch 移到 device(Dataloader返回的batch在cpu上，这一步将其放在GPU上)
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else
                {kk: vv.to(device) for kk, vv in v.items()} if isinstance(v, dict) else v
                for k, v in batch.items()
            }

            B = batch["actions"].shape[0]
            T = cfg.future_action_window_size + 1
            C = cfg.action_dim
            noise = torch.randn(B, T, C, device=device)

            # 2. 教师轨迹（带记录）
            with torch.no_grad():
                teacher_results = run_teacher_with_recording(
                    teacher, batch, noise,
                    num_steps=cfg.num_ddim_steps_teacher,
                    cfg_scale=cfg.cfg_scale_teacher,
                    device=device,
                )

            # 3. 学生：用教师输出的 z_corr 做 DDIM 采样
            z_corr = teacher_results["z_corr"]
            x0_student_corr = run_student_ddim(
                student, noise, z_corr, cfg.num_ddim_steps_student, device
            )

            # 4. 学生(z_wrong)，若做指令约束
            x0_student_wrong = None
            if cfg.use_instruction_constraint and B > 1:
                shuffle_idx = torch.randperm(B, device=device)
                batch_wrong = dict(batch)
                batch_wrong["input_ids"] = batch["input_ids"][shuffle_idx]
                batch_wrong["attention_mask"] = batch["attention_mask"][shuffle_idx]
                with torch.no_grad():
                    z_wrong = get_cognition_features(teacher, batch_wrong, device)
                x0_student_wrong = run_student_ddim(
                    student, noise, z_wrong, cfg.num_ddim_steps_student, device
                )

            # 5. Loss + backward
            loss_dict = compute_loss(
                teacher_results, x0_student_corr, x0_student_wrong, student, batch, cfg
            )
            optimizer.zero_grad()
            loss_dict["total"].backward()
            optimizer.step()

    # 6. 保存学生
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(student.state_dict(), cfg.output_dir / "student_final.pt")

