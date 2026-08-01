import os
# 在 train.py 或 distillation_cogact.py 最顶部，在任何 import 之前

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")  # 保持已有设置
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"   # TF 按需申请显存，而不是一次性占满
# 或者更彻底：os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import math
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, Any

import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
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
from distillation.runners import (
    get_student_timesteps,
    run_teacher_with_recording,
    run_student_ddim_with_recording,
)
from distillation.loss import compute_loss
from distillation.checkpoint import load_checkpoint, save_checkpoint, student_state_dict
from distillation.distributed_utils import get_dist_info
from distillation.log import (
    DistillationTrackers,
    append_metrics_jsonl,
    format_metric_line,
    log_memory,
)
from distillation.train_utils import TrainController, run_train_eval



overwatch = initialize_overwatch(__name__)

"""是主训练函数，用于训练学生模型。入口函数调用这个函数"""


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _grad_norm(parameters) -> Optional[float]:
    total_sq = 0.0
    found_grad = False
    for p in parameters:
        if p.grad is None:
            continue
        found_grad = True
        param_norm = p.grad.detach().data.norm(2)
        total_sq += float(param_norm.item() ** 2)
    if not found_grad:
        return None
    return total_sq ** 0.5


def _scheduled_lr(step: int, total_steps: Optional[int], base_lr: float, min_lr: float, warmup_ratio: float) -> float:
    if total_steps is None or total_steps <= 0:
        return base_lr
    warmup_steps = int(total_steps * warmup_ratio)
    if warmup_steps > 0 and step <= warmup_steps:
        return min_lr + (base_lr - min_lr) * (step / warmup_steps)
    decay_steps = max(total_steps - warmup_steps, 1)
    decay_progress = min(max((step - warmup_steps) / decay_steps, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    return min_lr + (base_lr - min_lr) * cosine


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def _checkpoint_name(prefix: str, suffix: str, step: Optional[int] = None) -> str:
    suffix_part = f"_{suffix}" if suffix else ""
    if step is None:
        return f"{prefix}{suffix_part}.pt"
    return f"{prefix}_{step:06d}{suffix_part}.pt"


def train_distillation(cfg: DistillationConfig, hf_token: Optional[str] = None) -> None:
    """
    主训练入口。
    流程：[FSDP 时先 set_device+empty_cache] → load_teacher → [FSDP wrap 或 to(device)] → load_student → ...
    与原版 scripts/train.py 的显存管理顺序对齐。
    """
    device: torch.device
    dist_info = get_dist_info()

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
    teacher_action_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in teacher.action_model.state_dict().items()
    }
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
        device = dist_info.device
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
        initial_state_dict=teacher_action_state,
    )
    del teacher_action_state
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

    if cfg.use_student_ddp and dist_info.world_size > 1:
        student.net = DDP(
            student.net,
            device_ids=[dist_info.local_rank],
            output_device=dist_info.local_rank,
            find_unused_parameters=True, # TODO 先从False改成True
            broadcast_buffers=False,
        )
        overwatch.info(
            f"Student action net wrapped with DDP | rank={dist_info.rank} "
            f"local_rank={dist_info.local_rank} world_size={dist_info.world_size}"
        )

    optimizer = torch.optim.AdamW(student.parameters(), lr=cfg.base_lr)
    resume_state = load_checkpoint(cfg.resume_checkpoint, student, optimizer, map_location=device)
    if resume_state["loaded"]:
        overwatch.info(
            "\n==========\n"
            f"DISTILL CHECKPOINT LOADED | path={resume_state['path']} "
            f"step={resume_state['step']} epoch={resume_state['epoch']}\n"
            "=========="
        )

    if cfg.log_memory:
        _sync_cuda()
        log_memory("before_train_loop", log_tf=cfg.log_memory_tf)

    global_batch_size = cfg.batch_size * dist_info.world_size
    overwatch.info(
        f"DISTILL BATCH CONFIG | per_device_batch_size={cfg.batch_size} "
        f"world_size={dist_info.world_size} global_batch_size={global_batch_size} "
        f"use_student_ddp={cfg.use_student_ddp and dist_info.world_size > 1}"
    )

    # E. 训练
    batch_count = int(resume_state["step"])
    stop_training = False
    metrics_path = cfg.output_dir / "metrics.jsonl"
    if overwatch.is_rank_zero():
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text("", encoding="utf-8")
    run_id = cfg.run_id or cfg.output_dir.name
    trackers = DistillationTrackers(
        trackers=tuple(cfg.trackers),
        run_id=run_id,
        output_dir=cfg.output_dir,
        hparams=draccus.encode(cfg),
        wandb_project=cfg.wandb_project,
        wandb_entity=cfg.wandb_entity,
        swanlab_project=cfg.swanlab_project,
        swanlab_workspace=cfg.swanlab_workspace,
        swanlab_mode=cfg.swanlab_mode,
        enabled=overwatch.is_rank_zero(),
    )
    controller = TrainController(
        eval_interval=cfg.train_eval_interval,
        patience=cfg.early_stop_patience,
        min_delta=cfg.early_stop_min_delta,
        metric_name=cfg.early_stop_metric,
        output_dir=cfg.output_dir,
        checkpoint_suffix=cfg.checkpoint_suffix,
        enabled=cfg.enable_train_eval,
    )
    for epoch in range(cfg.epochs):
        sampler = getattr(dataloader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        for batch in dataloader:
            if cfg.max_batches is not None and batch_count >= cfg.max_batches:
                overwatch.info(
                    "\n==========\n"
                    f"DISTILL STOP | reached max_batches={cfg.max_batches}\n"
                    "=========="
                )
                stop_training = True
                break  # 提前停下，以供debug，但仍走统一保存 checkpoint
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
            actions_future = batch["actions"][:, -T:, :]
            expected_actions_shape = (B, T, C)
            if tuple(actions_future.shape) != expected_actions_shape:
                raise RuntimeError(
                    "actions_future shape mismatch: "
                    f"got={tuple(actions_future.shape)}, expected={expected_actions_shape}"
                )
            noise = torch.randn(B, T, C, device=device)
            student_timesteps = get_student_timesteps(student, cfg.num_ddim_steps_student)

            # 2. Teacher DDIM: record both timestep-aligned samples and the full depth path.
    
            with torch.no_grad():
                teacher_results = run_teacher_with_recording(
                    teacher, batch, noise,
                    num_steps=cfg.num_ddim_steps_teacher,
                    record_timesteps=student_timesteps,
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
                    f"z_corr={z_corr_shape} "
                    f"x0_teacher={tuple(teacher_results['x0_teacher'].shape)}"
                )

            if cfg.log_memory and first_step:
                _sync_cuda()
                log_memory("train_step1_after_teacher_recording", log_tf=cfg.log_memory_tf)

            # 3. 学生：用教师输出的 z_corr 做 DDIM 采样
            z_corr = teacher_results["z_corr"]
            x0_teacher = teacher_results["x0_teacher"]
            student_results = run_student_ddim_with_recording(
                student, noise, z_corr, cfg.num_ddim_steps_student, device
            )
            x0_student = student_results["x0_student"]
            if first_step:
                overwatch.info(
                    f"=================================================================="
                    f"teacher_full_timesteps={teacher_results['teacher_full_timesteps']} "
                    f"student_timesteps={student_timesteps} "
                    f"teacher_recorded_timesteps={teacher_results['timesteps']} "
                    f"student_recorded_timesteps={student_results['timesteps']} "
                    f"teacher_trajectory.shape={list(teacher_results['trajectory'].shape)} "
                    f"student_trajectory.shape={list(student_results['trajectory'].shape)} "
                    f"teacher_depth_path.shape={list(teacher_results['depth_x0_path'].shape)} "
                    f"student_depth_path.shape={list(student_results['depth_x0_path'].shape)}"
                )
            if x0_teacher.shape != x0_student.shape or x0_student.shape != actions_future.shape:
                raise RuntimeError(
                    "distillation output shape mismatch: "
                    f"actions_future={tuple(actions_future.shape)}, "
                    f"z_corr={tuple(z_corr.shape)}, "
                    f"x0_teacher={tuple(x0_teacher.shape)}, "
                    f"x0_student={tuple(x0_student.shape)}"
                )

            if cfg.log_memory and first_step:
                _sync_cuda()
                log_memory("train_step1_after_student_ddim_corr", log_tf=cfg.log_memory_tf)

            # 5. Loss + backward
            loss_dict = compute_loss(
                student=student,
                actions_future=actions_future,
                z_corr=z_corr,
                x0_teacher=x0_teacher,
                x0_student=x0_student,
                teacher_trajectory=teacher_results["trajectory"],
                student_trajectory=student_results["trajectory"],
                cfg=cfg,
                teacher_depth_path=teacher_results["depth_x0_path"],
                student_depth_path=student_results["depth_x0_path"],
            )
            for loss_name, loss_value in loss_dict.items():
                if not torch.isfinite(loss_value):
                    raise RuntimeError(
                        f"non-finite distillation loss: {loss_name}={loss_value.item()} "
                        f"at step={batch_count}"
                    )
            if cfg.log_memory and first_step:
                _sync_cuda()
                log_memory("train_step1_before_backward", log_tf=cfg.log_memory_tf)
            lr = _scheduled_lr(
                batch_count,
                cfg.max_batches,
                cfg.base_lr,
                cfg.min_lr,
                cfg.warmup_ratio,
            )
            _set_optimizer_lr(optimizer, lr)
            optimizer.zero_grad()
            loss_dict["total"].backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.max_grad_norm).item())
            was_clipped = grad_norm > cfg.max_grad_norm
            optimizer.step()
            loss_total = float(loss_dict["total"].item())
            loss_task = float(loss_dict["task"].item())
            loss_final_gt = float(loss_dict["final_gt"].item())
            loss_final_teacher = float(loss_dict["final_teacher"].item())
            loss_traj_teacher = float(loss_dict["traj_teacher"].item())
            loss_path_teacher = float(loss_dict["path_teacher"].item())
            loss_macro_teacher = float(loss_dict["macro_teacher"].item())
            cuda_memory_allocated_mb = (
                float(torch.cuda.memory_allocated(device) / 1024**2) if torch.cuda.is_available() else 0.0
            )
            cuda_max_memory_allocated_mb = (
                float(torch.cuda.max_memory_allocated(device) / 1024**2) if torch.cuda.is_available() else 0.0
            )
            metrics = {
                "step": batch_count,
                "epoch": epoch + 1,
                "loss_total": loss_total,
                "loss_task": loss_task,
                "loss_final_gt": loss_final_gt,
                "loss_final_teacher": loss_final_teacher,
                "loss_traj_teacher": loss_traj_teacher,
                "loss_path_teacher": loss_path_teacher,
                "loss_macro_teacher": loss_macro_teacher,
                "lambda_task": cfg.lambda_task,
                "lambda_final": cfg.lambda_final,
                "lambda_final_gt": cfg.lambda_final_gt,
                "lambda_traj": cfg.lambda_traj,
                "lambda_path": cfg.lambda_path,
                "lambda_macro": cfg.lambda_macro,
                "lambda_neg": cfg.lambda_neg,
                "grad_norm": grad_norm,
                "max_grad_norm": cfg.max_grad_norm,
                "was_clipped": was_clipped,
                "lr": lr,
                "actions_shape": list(actions_future.shape),
                "z_corr_shape": list(z_corr.shape),
                "x0_teacher_shape": list(x0_teacher.shape),
                "x0_student_shape": list(x0_student.shape),
                "student_trajectory_shape": list(student_results["trajectory"].shape),
                "student_timesteps": student_results["timesteps"],
                "teacher_trajectory_shape": list(teacher_results["trajectory"].shape),
                "teacher_timesteps": teacher_results["timesteps"],
                "teacher_full_timesteps": teacher_results["teacher_full_timesteps"],
                "teacher_depth_path_shape": list(teacher_results["depth_x0_path"].shape),
                "student_depth_path_shape": list(student_results["depth_x0_path"].shape),
                "teacher_depth_node_count": int(teacher_results["depth_x0_path"].shape[0]),
                "student_depth_node_count": int(student_results["depth_x0_path"].shape[0]),
                "cuda_memory_allocated_mb": cuda_memory_allocated_mb,
                "cuda_max_memory_allocated_mb": cuda_max_memory_allocated_mb,
            }
            if overwatch.is_rank_zero():
                append_metrics_jsonl(metrics_path, metrics)
                trackers.log(
                    {
                        "epoch": epoch + 1,
                        "loss/total": loss_total,
                        "loss/task": loss_task,
                        "loss/final_gt": loss_final_gt,
                        "loss/final_teacher": loss_final_teacher,
                        "loss/traj_teacher": loss_traj_teacher,
                        "loss/path_teacher": loss_path_teacher,
                        "loss/macro_teacher": loss_macro_teacher,
                        "lambda/task": cfg.lambda_task,
                        "lambda/final": cfg.lambda_final,
                        "lambda/final_gt": cfg.lambda_final_gt,
                        "lambda/traj": cfg.lambda_traj,
                        "lambda/path": cfg.lambda_path,
                        "lambda/macro": cfg.lambda_macro,
                        "lambda/neg": cfg.lambda_neg,
                        "depth/teacher_node_count": metrics["teacher_depth_node_count"],
                        "depth/student_node_count": metrics["student_depth_node_count"],
                        "memory/cuda_allocated_mb": cuda_memory_allocated_mb,
                        "memory/cuda_max_allocated_mb": cuda_max_memory_allocated_mb,
                        "grad_norm": grad_norm,
                        "max_grad_norm": cfg.max_grad_norm,
                        "was_clipped": float(was_clipped),
                        "lr": lr,
                    },
                    step=batch_count,
                )
            metric_line = format_metric_line(
                step=batch_count,
                epoch=epoch + 1,
                epochs=cfg.epochs,
                loss_total=loss_total,
                loss_task=loss_task,
                loss_final_gt=loss_final_gt,
                loss_final_teacher=loss_final_teacher,
                loss_traj_teacher=loss_traj_teacher,
                loss_path_teacher=loss_path_teacher,
                loss_macro_teacher=loss_macro_teacher,
                grad_norm=grad_norm,
                lr=lr,
            )
            if overwatch.is_rank_zero() and (batch_count == 1 or batch_count % 20 == 0):
                print(metric_line, flush=True)
            should_checkpoint = cfg.checkpoint_interval > 0 and batch_count % cfg.checkpoint_interval == 0
            should_eval = controller.should_eval(batch_count)
            step_checkpoint_path = None
            if should_checkpoint or should_eval:
                step_checkpoint_path = cfg.output_dir / _checkpoint_name(
                    "checkpoint_step",
                    cfg.checkpoint_suffix,
                    step=batch_count,
                )
                save_checkpoint(
                    step_checkpoint_path,
                    student=student,
                    optimizer=optimizer,
                    epoch=epoch,
                    step=batch_count,
                    cfg=cfg,
                )
                overwatch.info(f"DISTILL CHECKPOINT SAVED | path={step_checkpoint_path}")

            if should_eval and step_checkpoint_path is not None:
                if dist.is_available() and dist.is_initialized():
                    dist.barrier()
                eval_summary = run_train_eval(
                    teacher=teacher,
                    student=student,
                    cfg=cfg,
                    device=device,
                    max_eval_batches=cfg.train_eval_batches,
                )
                stop_value = 0
                if overwatch.is_rank_zero():
                    eval_summary = {
                        **eval_summary,
                        "step": batch_count,
                        "checkpoint": str(step_checkpoint_path),
                    }
                    append_metrics_jsonl(metrics_path, eval_summary)
                    trackers.log(
                        {
                            "eval/mean_loss_final_gt": eval_summary["eval/mean_loss_final_gt"],
                            "eval/mean_loss_final_teacher": eval_summary["eval/mean_loss_final_teacher"],
                            "eval/mean_loss_traj_teacher": eval_summary["eval/mean_loss_traj_teacher"],
                            "eval/mean_loss_path_teacher": eval_summary["eval/mean_loss_path_teacher"],
                            "eval/mean_loss_macro_teacher": eval_summary["eval/mean_loss_macro_teacher"],
                            "eval/mean_student_teacher_final_mse": eval_summary[
                                "eval/mean_student_teacher_final_mse"
                            ],
                            "eval/mean_student_gt_action_mse": eval_summary[
                                "eval/mean_student_gt_action_mse"
                            ],
                            "eval/mean_teacher_gt_action_mse": eval_summary[
                                "eval/mean_teacher_gt_action_mse"
                            ],
                        },
                        step=batch_count,
                    )
                    stop_value = int(
                        controller.update(
                            step=batch_count,
                            checkpoint_path=step_checkpoint_path,
                            eval_summary=eval_summary,
                        )
                    )
                    overwatch.info(
                        "DISTILL TRAIN EVAL | "
                        f"step={batch_count} "
                        f"{cfg.early_stop_metric}={eval_summary[cfg.early_stop_metric]:.6f} "
                        f"best={controller.best_metric:.6f} "
                        f"bad_eval_count={controller.bad_eval_count}/{controller.patience} "
                        f"best_checkpoint={controller.best_checkpoint_path}"
                    )
                if dist.is_available() and dist.is_initialized():
                    stop_tensor = torch.tensor([stop_value], device=device, dtype=torch.int64)
                    dist.broadcast(stop_tensor, src=0)
                    stop_value = int(stop_tensor.item())
                if stop_value:
                    overwatch.info(
                        "\n==========\n"
                        f"DISTILL STOP | early stopping at step={batch_count}\n"
                        "=========="
                    )
                    stop_training = True
                    break

            if cfg.log_memory and first_step:
                _sync_cuda()
                log_memory("train_step1_after_optimizer_step", log_tf=cfg.log_memory_tf)
        if stop_training:
            break

    # 6. 保存学生
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    student_path = cfg.output_dir / _checkpoint_name("student_final", cfg.checkpoint_suffix)
    final_checkpoint_path = cfg.output_dir / _checkpoint_name("checkpoint_final", cfg.checkpoint_suffix)
    final_source = None
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    if overwatch.is_rank_zero():
        if cfg.enable_train_eval and controller.best_checkpoint_alias().exists():
            best_checkpoint = torch.load(
                controller.best_checkpoint_alias(),
                map_location="cpu",
                weights_only=True,
            )
            torch.save(best_checkpoint["student"], student_path)
            torch.save(best_checkpoint, final_checkpoint_path)
            final_source = controller.best_checkpoint_alias()
        else:
            torch.save(student_state_dict(student), student_path)
            save_checkpoint(
                final_checkpoint_path,
                student=student,
                optimizer=optimizer,
                epoch=epoch,
                step=batch_count,
                cfg=cfg,
            )
    overwatch.info(
        "\n==========\n"
        f"DISTILL CHECKPOINT SAVED | student_path={student_path} checkpoint_path={final_checkpoint_path} "
        f"source={final_source or 'last'}\n"
        "=========="
    )
    trackers.finalize()
