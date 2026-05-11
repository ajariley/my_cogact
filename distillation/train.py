import os
# 在 train.py 或 distillation_cogact.py 最顶部，在任何 import 之前

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")  # 保持已有设置
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"   # TF 按需申请显存，而不是一次性占满
# 或者更彻底：os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, Any

import torch
from torch.utils.data import DataLoader
import draccus
from torch.cuda.amp import autocast

# 分布式
import torch.distributed as dist

# 必须在 initialize_overwatch() → PartialState → init_process_group 之前绑定本进程 GPU，
# 否则 NCCL 会报 “device used by this process is currently unknown”。
if torch.cuda.is_available():
    _lr = os.environ.get("LOCAL_RANK")
    if _lr is not None:
        torch.cuda.set_device(int(_lr))

from prismatic.overwatch import initialize_overwatch

from vla import load_vla, CogACT
from action_model.action_model import ActionModel
from prismatic.vla import get_vla_dataset_and_collator

from conf.distillation import DistillationConfig
from distillation.loaders import load_teacher, load_student, load_dataloader
from distillation.runners import run_teacher_with_recording, run_student_ddim, get_cognition_features
from distillation.loss import compute_loss
from distillation.memory_log import log_memory



overwatch = initialize_overwatch(__name__)

"""是主训练函数，用于训练学生模型。入口函数调用这个函数"""


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def train_distillation(cfg: DistillationConfig, hf_token: Optional[str] = None) -> None:
    """
    主训练入口。
    流程：[FSDP 时先 set_device+empty_cache] → load_teacher → [FSDP wrap 或 to(device)] → load_student → ...
    与原版 scripts/train.py 的显存管理顺序对齐。
    """
    device: torch.device

    # Support both --teacher_checkpoint and --pretrained_checkpoint CLI flags.
    checkpoint_input = cfg.pretrained_checkpoint or cfg.teacher_checkpoint
    checkpoint_path = Path(checkpoint_input)
    if checkpoint_path.is_dir():
        # Allow passing run dir, e.g. /path/to/CogACT-Base
        candidate = checkpoint_path / "checkpoints" / "CogACT-Base.pt"
        if candidate.exists():
            checkpoint_path = candidate
        else:
            pt_files = sorted((checkpoint_path / "checkpoints").glob("*.pt"))
            if not pt_files:
                raise FileNotFoundError(
                    f"No checkpoint .pt found under directory: {checkpoint_path / 'checkpoints'}"
                )
            checkpoint_path = pt_files[-1]

    # A. FSDP 模式：先设置设备、清空显存（与 scripts/train.py:131-132 对齐）
    if cfg.use_fsdp:
        if int(os.environ.get("WORLD_SIZE", -1)) <= 0:
            raise RuntimeError(
                "use_fsdp=True 需 torchrun 启动，请使用: "
                "torchrun --standalone --nnodes 1 --nproc-per-node 1 scripts/distillation_cogact.py --use_fsdp True ..."
            )
        torch.cuda.set_device(overwatch.local_rank())
        torch.cuda.empty_cache()
        device = torch.device("cuda", overwatch.local_rank())
        if cfg.log_memory:
            log_memory("after_cuda_init_fsdp", log_tf=cfg.log_memory_tf)

    # B. 加载教师，并强制置于 CPU（避免 HF inference 等路径将模型加载到 GPU，导致 FSDP init OOM）
    with autocast(dtype=torch.bfloat16):
        teacher = load_teacher(
            checkpoint_path,
            cfg.action_model_type_teacher,
            cfg.future_action_window_size,
            hf_token=hf_token,
        )
    if cfg.log_memory:
        log_memory("after_load_teacher", log_tf=cfg.log_memory_tf)
    teacher = teacher.cpu()
    if cfg.log_memory:
        log_memory("after_teacher_cpu", log_tf=cfg.log_memory_tf)

    # C. FSDP 包装或移至 device（在 load_student 之前）
    if cfg.use_fsdp:
        torch.cuda.empty_cache()  # wrap 前再清一次，减少碎片
        from distillation.fsdp import wrap_teacher_with_fsdp

        teacher = wrap_teacher_with_fsdp(
            teacher,
            use_cpu_offload=cfg.use_cpu_offload,
            use_bf16=cfg.use_bf16,
            reduce_in_full_precision=cfg.reduce_in_full_precision,
        )
        # Explicitly bind barrier to this rank's GPU to avoid NCCL "unknown device" warning.
        dist.barrier(device_ids=[overwatch.local_rank()])
        if cfg.log_memory:
            log_memory("after_teacher_on_device", log_tf=cfg.log_memory_tf)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if cfg.use_bf16:
            teacher.vlm = teacher.vlm.to(torch.bfloat16)
        teacher = teacher.to(device)
        if cfg.log_memory:
            log_memory("after_teacher_on_device", log_tf=cfg.log_memory_tf)

    # D. 加载学生、DataLoader、optimizer
    token_size = teacher.llm_backbone.llm.lm_head.in_features  # 4096
    student = load_student(
        token_size=token_size,
        action_model_type=cfg.action_model_type_student,
        in_channels=cfg.action_dim,
        future_action_window_size=cfg.future_action_window_size,
        past_action_window_size=cfg.past_action_window_size,
        device=device,
    )
    if cfg.log_memory:
        log_memory("after_load_student", log_tf=cfg.log_memory_tf)

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

    if cfg.log_memory:
        _sync_cuda()
        log_memory("before_train_loop", log_tf=cfg.log_memory_tf)

    # E. 训练
    batch_count = 0
    for epoch in range(cfg.epochs):
        for batch in dataloader:
            if cfg.max_batches is not None and batch_count >= cfg.max_batches:
                overwatch.info(
                    "\n==========\n"
                    f"DISTILL STOP | reached max_batches={cfg.max_batches}\n"
                    "=========="
                )
                return  # 提前停下，以供debug
            batch_count += 1
            first_step = batch_count == 1

            # 将 batch 移到 device(Dataloader返回的batch在cpu上，这一步将其放在GPU上)
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else
                {kk: vv.to(device) for kk, vv in v.items()} if isinstance(v, dict) else v
                for k, v in batch.items()
            }

            if cfg.log_memory and first_step:
                _sync_cuda()
                log_memory("train_step1_after_batch_to_device", log_tf=cfg.log_memory_tf)

            B = batch["actions"].shape[0]
            T = cfg.future_action_window_size + 1
            C = cfg.action_dim
            noise = torch.randn(B, T, C, device=device)

            # 2. 教师轨迹（带记录）
            if cfg.log_memory and first_step:
                overwatch.info(
                    f"[distill_debug] step={batch_count} rank={overwatch.rank()} "
                    f"before_run_teacher_with_recording "
                    f"input_ids={tuple(batch['input_ids'].shape)} "
                    f"attention_mask={tuple(batch['attention_mask'].shape)} "
                    f"actions={tuple(batch['actions'].shape)} "
                    f"noise={tuple(noise.shape)}"
                )
            with torch.no_grad():
                teacher_results = run_teacher_with_recording(
                    teacher, batch, noise,
                    num_steps=cfg.num_ddim_steps_teacher,
                    cfg_scale=cfg.cfg_scale_teacher,
                    device=device,
                )
            if cfg.log_memory and first_step:
                z_corr_shape = (
                    tuple(teacher_results["z_corr"].shape)
                    if isinstance(teacher_results.get("z_corr"), torch.Tensor)
                    else "N/A"
                )
                overwatch.info(
                    f"[distill_debug] step={batch_count} rank={overwatch.rank()} "
                    f"after_run_teacher_with_recording "
                    f"z_corr={z_corr_shape}"
                )

            if cfg.log_memory and first_step:
                _sync_cuda()
                log_memory("train_step1_after_teacher_recording", log_tf=cfg.log_memory_tf)

            # 3. 学生：用教师输出的 z_corr 做 DDIM 采样
            z_corr = teacher_results["z_corr"]
            x0_student_corr = run_student_ddim(
                student, noise, z_corr, cfg.num_ddim_steps_student, device
            )

            if cfg.log_memory and first_step:
                _sync_cuda()
                log_memory("train_step1_after_student_ddim_corr", log_tf=cfg.log_memory_tf)

            # 4. 学生(z_wrong)，若做指令约束
            x0_student_wrong = None
            if cfg.use_instruction_constraint and B > 1:
                shuffle_idx = torch.randperm(B, device=device)
                batch_wrong = dict(batch)
                batch_wrong["input_ids"] = batch["input_ids"][shuffle_idx]
                batch_wrong["attention_mask"] = batch["attention_mask"][shuffle_idx]
                if cfg.log_memory and first_step:
                    overwatch.info(
                        f"[distill_debug] step={batch_count} rank={overwatch.rank()} "
                        f"before_get_cognition_features_wrong "
                        f"shuffle_idx={tuple(shuffle_idx.shape)} "
                        f"wrong_input_ids={tuple(batch_wrong['input_ids'].shape)} "
                        f"wrong_attention_mask={tuple(batch_wrong['attention_mask'].shape)}"
                    )
                with torch.no_grad():
                    z_wrong = get_cognition_features(teacher, batch_wrong, device)
                if cfg.log_memory and first_step:
                    overwatch.info(
                        f"[distill_debug] step={batch_count} rank={overwatch.rank()} "
                        f"after_get_cognition_features_wrong "
                        f"z_wrong={tuple(z_wrong.shape)}"
                    )
                x0_student_wrong = run_student_ddim(
                    student, noise, z_wrong, cfg.num_ddim_steps_student, device
                )

            if cfg.log_memory and first_step:
                _sync_cuda()
                log_memory("train_step1_after_student_ddim_wrong_if_any", log_tf=cfg.log_memory_tf)

            # 5. Loss + backward
            loss_dict = compute_loss(
                teacher_results, x0_student_corr, x0_student_wrong, student, batch, cfg
            )
            if cfg.log_memory and first_step:
                _sync_cuda()
                log_memory("train_step1_before_backward", log_tf=cfg.log_memory_tf)
            optimizer.zero_grad()
            loss_dict["total"].backward()
            optimizer.step()
            overwatch.info(
                "\n==========\n"
                "DISTILL STEP DONE\n"
                f"epoch={epoch + 1}/{cfg.epochs} step={batch_count} "
                f"loss={loss_dict['total'].item():.6f}\n"
                "=========="
            )

            if cfg.log_memory and first_step:
                _sync_cuda()
                log_memory("train_step1_after_optimizer_step", log_tf=cfg.log_memory_tf)

    # 6. 保存学生
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(student.state_dict(), cfg.output_dir / "student_final.pt")
