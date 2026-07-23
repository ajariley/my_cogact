"""
Evaluate a distilled CogACT action head against its teacher.

Example:
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 python scripts/eval_distillation.py \
      --use_fsdp False \
      --data_root_dir "/home/huangjiaqi/.vscode-server/openvla_data" \
      --pretrained_checkpoint "/home/huangjiaqi/projects/CogACT-Base" \
      --resume_checkpoint runs/distillation/checkpoint_final.pt \
      --data_mix bridge \
      --batch_size 1 \
      --max_batches 10 \
      --num_ddim_steps_teacher 20 \
      --num_ddim_steps_student 4
"""

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

import draccus
import torch
import torch.distributed as dist
from torch.cuda.amp import autocast

from action_model.action_model import ActionModel
from conf import DistillationConfig
from distillation.checkpoint import load_checkpoint
from distillation.loaders import load_dataloader, load_student, load_teacher
from distillation.runners import (
    get_student_timesteps,
    run_student_ddim_with_recording,
    run_teacher_with_recording,
)
from distillation.path import depth_path_losses
from prismatic.overwatch import initialize_overwatch


overwatch = initialize_overwatch(__name__)


# 解析 teacher checkpoint，支持直接传 .pt，也支持传 CogACT-Base 目录。
def _resolve_teacher_checkpoint(cfg: DistillationConfig) -> Path:
    checkpoint_input = cfg.pretrained_checkpoint or cfg.teacher_checkpoint
    checkpoint_path = Path(checkpoint_input)
    if checkpoint_path.is_dir():
        candidate = checkpoint_path / "checkpoints" / "CogACT-Base.pt"
        if candidate.exists():
            return candidate
        pt_files = sorted((checkpoint_path / "checkpoints").glob("*.pt"))
        if pt_files:
            return pt_files[-1]
        raise FileNotFoundError(f"No checkpoint .pt found under directory: {checkpoint_path / 'checkpoints'}")
    return checkpoint_path


# 将 dataloader 给出的 batch 递归搬到当前 eval device。
def _move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else
        {kk: vv.to(device) for kk, vv in v.items()} if isinstance(v, dict) else v
        for k, v in batch.items()
    }


# 从 batch 中取当前动作窗口，并校验它和训练时的 [B, T, C] 约定一致。
def _actions_future(batch: Dict[str, Any], cfg: DistillationConfig) -> torch.Tensor:
    batch_size = batch["actions"].shape[0]
    horizon = cfg.future_action_window_size + 1
    expected_shape = (batch_size, horizon, cfg.action_dim)
    actions = batch["actions"][:, -horizon:, :]
    if tuple(actions.shape) != expected_shape:
        raise RuntimeError(f"actions_future shape mismatch: got={tuple(actions.shape)}, expected={expected_shape}")
    return actions


# eval 指标采用 jsonl 追加写入，便于逐 batch 查看和后处理。
def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _batch_to_cpu(batch: Dict[str, Any]) -> Dict[str, Any]:
    return {
        k: v.detach().cpu() if isinstance(v, torch.Tensor) else
        {kk: vv.detach().cpu() for kk, vv in v.items()} if isinstance(v, dict) else v
        for k, v in batch.items()
    }


def _build_eval_cache(
    *,
    dataloader,
    cfg: DistillationConfig,
    device: torch.device,
    max_eval_batches: int,
) -> Dict[str, Any]:
    torch.manual_seed(cfg.eval_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.eval_seed)

    items = []
    for count, batch in enumerate(dataloader, start=1):
        if count > max_eval_batches:
            break
        batch = _move_batch_to_device(batch, device)
        actions_future = _actions_future(batch, cfg)
        noise = torch.randn_like(actions_future)
        items.append({"batch": _batch_to_cpu(batch), "noise": noise.detach().cpu()})
    if not items:
        raise RuntimeError("eval cache produced zero batches")
    return {
        "eval_seed": cfg.eval_seed,
        "batch_size": cfg.batch_size,
        "max_batches": max_eval_batches,
        "future_action_window_size": cfg.future_action_window_size,
        "action_dim": cfg.action_dim,
        "items": items,
    }

# 如果已经存在了，就不再生成
def _load_or_build_eval_cache(
    *,
    dataloader,
    cfg: DistillationConfig,
    device: torch.device,
    max_eval_batches: int,
) -> tuple[Dict[str, Any], Path]:
    cache_path = cfg.eval_cache_path or (cfg.output_dir / f"eval_cache_seed{cfg.eval_seed}_b{cfg.batch_size}_n{max_eval_batches}.pt")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        if len(cache.get("items", [])) != max_eval_batches:
            overwatch.info(
                f"EVAL CACHE REBUILD | path={cache_path} "
                f"cached_batches={len(cache.get('items', []))} requested_batches={max_eval_batches}"
            )
        else:
            return cache, cache_path
    cache = _build_eval_cache(
        dataloader=dataloader,
        cfg=cfg,
        device=device,
        max_eval_batches=max_eval_batches,
    )
    torch.save(cache, cache_path)
    return cache, cache_path


# 单个 batch 的核心 eval：共享 noise，跑 teacher/student trajectory，并计算最终点和轨迹 MSE。
def _eval_one_batch(
    *,
    teacher,
    student: ActionModel,
    batch: Dict[str, Any],
    noise: torch.Tensor,
    cfg: DistillationConfig,
    device: torch.device,
) -> Dict[str, Any]:
    actions_future = _actions_future(batch, cfg)
    noise = noise.to(device=device, dtype=actions_future.dtype)
    student_timesteps = get_student_timesteps(student, cfg.num_ddim_steps_student)

    with torch.no_grad():
        teacher_out = run_teacher_with_recording(
            teacher,
            batch,
            noise,
            num_steps=cfg.num_ddim_steps_teacher,
            record_timesteps=student_timesteps,
            cfg_scale=cfg.cfg_scale_teacher,
            device=device,
        )
        student_out = run_student_ddim_with_recording(
            student,
            noise,
            teacher_out["z_corr"],
            cfg.num_ddim_steps_student,
            device,
        )

    x0_teacher = teacher_out["x0_teacher"]
    x0_student = student_out["x0_student"]
    teacher_traj = teacher_out["trajectory"]
    student_traj = student_out["trajectory"]

    if x0_teacher.shape != x0_student.shape or x0_student.shape != actions_future.shape:
        raise RuntimeError(
            "eval output shape mismatch: "
            f"actions_future={tuple(actions_future.shape)}, "
            f"x0_teacher={tuple(x0_teacher.shape)}, "
            f"x0_student={tuple(x0_student.shape)}"
        )
    if teacher_traj.shape != student_traj.shape:
        raise RuntimeError(
            "eval trajectory shape mismatch: "
            f"teacher={tuple(teacher_traj.shape)}, student={tuple(student_traj.shape)}"
        )

    loss_final = torch.nn.functional.mse_loss(x0_student, x0_teacher.detach())
    loss_traj = torch.nn.functional.mse_loss(student_traj, teacher_traj.detach())
    loss_path, loss_macro, _ = depth_path_losses(
        student_out["depth_x0_path"],
        teacher_out["depth_x0_path"],
        action_dim_weights=cfg.refinement_action_dim_weights,
        eps=cfg.refinement_progress_eps,
    )
    student_gt_action_mse = torch.nn.functional.mse_loss(x0_student, actions_future)
    teacher_gt_action_mse = torch.nn.functional.mse_loss(x0_teacher, actions_future)
    return {
        "eval/loss_final": float(loss_final.item()),
        "eval/loss_traj": float(loss_traj.item()),
        "eval/loss_path": float(loss_path.item()),
        "eval/loss_macro": float(loss_macro.item()),
        "eval/student_teacher_final_mse": float(loss_final.item()),
        "eval/student_gt_action_mse": float(student_gt_action_mse.item()),
        "eval/teacher_gt_action_mse": float(teacher_gt_action_mse.item()),
        "teacher_timesteps": teacher_out["timesteps"],
        "student_timesteps": student_out["timesteps"],
        "teacher_full_timesteps": teacher_out["teacher_full_timesteps"],
        "teacher_trajectory_shape": list(teacher_traj.shape),
        "student_trajectory_shape": list(student_traj.shape),
        "teacher_depth_path_shape": list(teacher_out["depth_x0_path"].shape),
        "student_depth_path_shape": list(student_out["depth_x0_path"].shape),
        "x0_teacher_shape": list(x0_teacher.shape),
        "x0_student_shape": list(x0_student.shape),
    }


# eval 入口：加载模型与数据，循环有限个 batch，最后写入均值 summary。
@draccus.wrap()
def main(cfg: DistillationConfig) -> None:
    if cfg.resume_checkpoint is None:
        raise ValueError("eval requires --resume_checkpoint pointing to checkpoint_final.pt or checkpoint_step_XXXXXX.pt")

    torch.manual_seed(cfg.eval_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.eval_seed)

    # 1. eval 只做单进程/单卡推理，device 直接按当前 CUDA 可用性选择。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_checkpoint = _resolve_teacher_checkpoint(cfg)
    # teacher 用原始 CogACT checkpoint 加载，并保持 eval 模式作为固定监督源。
    with autocast(dtype=torch.bfloat16):
        teacher = load_teacher(
            teacher_checkpoint,
            cfg.action_model_type_teacher,
            cfg.future_action_window_size,
            hf_token="hf_xxx",
        )
    teacher = teacher.to(device)
    teacher.eval()

    # 2. student 结构按当前 cfg 创建，再从蒸馏 checkpoint 恢复权重。
    token_size = teacher.llm_backbone.llm.lm_head.in_features
    student = load_student(
        token_size=token_size,
        action_model_type=cfg.action_model_type_student,
        in_channels=cfg.action_dim,
        future_action_window_size=cfg.future_action_window_size,
        past_action_window_size=cfg.past_action_window_size,
        device=device,
    )
    load_checkpoint(cfg.resume_checkpoint, student, optimizer=None, map_location=device)
    overwatch.info(f"EVAL STUDENT CHECKPOINT LOADED | path={cfg.resume_checkpoint}")
    student.eval()

    # 3. max_batches 在 eval 中表示最多评估多少个 batch；不传时默认取 100 个。
    max_eval_batches = cfg.max_batches if cfg.max_batches is not None else 100

    # 3. eval 复用训练的数据管线，但关闭 image_aug，保证观察更稳定。
    dataloader = load_dataloader(
        teacher,
        cfg.data_root_dir,
        cfg.data_mix,
        cfg.batch_size,
        shuffle_buffer_size=max_eval_batches * cfg.batch_size,
        train=False,
        image_aug=False,
        load_all_data_for_training=cfg.load_all_data_for_training,
        future_action_window_size=cfg.future_action_window_size,
        past_action_window_size=cfg.past_action_window_size,
    )

    eval_path = cfg.output_dir / "eval_metrics.jsonl"
    eval_path.parent.mkdir(parents=True, exist_ok=True)

    eval_cache, eval_cache_path = _load_or_build_eval_cache(
        dataloader=dataloader,
        cfg=cfg,
        device=device,
        max_eval_batches=max_eval_batches,
    )
    overwatch.info(f"EVAL CACHE READY | path={eval_cache_path} num_batches={len(eval_cache['items'])}")
    sums = {
        "eval/loss_final": 0.0,
        "eval/loss_traj": 0.0,
        "eval/loss_path": 0.0,
        "eval/loss_macro": 0.0,
        "eval/student_teacher_final_mse": 0.0,
        "eval/student_gt_action_mse": 0.0,
        "eval/teacher_gt_action_mse": 0.0,
    }
    count = 0
    condition = {
        "type": "eval_conditions",
        "checkpoint": str(cfg.resume_checkpoint),
        "eval_seed": cfg.eval_seed,
        "lambda_task": cfg.lambda_task,
        "lambda_final": cfg.lambda_final,
        "lambda_traj": cfg.lambda_traj,
        "lambda_path": cfg.lambda_path,
        "lambda_macro": cfg.lambda_macro,
        "refinement_action_dim_weights": cfg.refinement_action_dim_weights,
        "lambda_neg": cfg.lambda_neg,
        "batch_size": cfg.batch_size,
        "max_batches": max_eval_batches,
        "num_ddim_steps_teacher": cfg.num_ddim_steps_teacher,
        "num_ddim_steps_student": cfg.num_ddim_steps_student,
        "shuffle_buffer_size":max_eval_batches * cfg.batch_size,
        "dataset_train": False,
        "eval_cache_path": str(eval_cache_path),
    }
    torch.manual_seed(cfg.eval_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.eval_seed)

    # 5. 每个 checkpoint 都复用同一份 eval cache，不再从 dataloader 重新取样本或 noise。
    for item in eval_cache["items"]:
        count += 1
        batch = _move_batch_to_device(item["batch"], device)
        row = _eval_one_batch(
            teacher=teacher,
            student=student,
            batch=batch,
            noise=item["noise"],
            cfg=cfg,
            device=device,
        )
        row = {
            "eval_step": count,
            "checkpoint": str(cfg.resume_checkpoint),
            **row,
        }
        for key in sums:
            sums[key] += row[key]
        print(
            f"eval_step={count} "
            f"loss_final={row['eval/loss_final']:.6f} "
            f"loss_traj={row['eval/loss_traj']:.6f} "
            f"loss_path={row['eval/loss_path']:.6f} "
            f"loss_macro={row['eval/loss_macro']:.6f} "
            f"student_teacher_final_mse={row['eval/student_teacher_final_mse']:.6f} "
            f"student_gt_action_mse={row['eval/student_gt_action_mse']:.6f} "
            f"teacher_gt_action_mse={row['eval/teacher_gt_action_mse']:.6f}",
            flush=True,
        )

    if count == 0:
        raise RuntimeError("eval produced zero batches")

    # 6. 最后一行写 summary，方便直接读取整体 eval 结果。
    summary = {
        "type": "eval_summary",
        "eval/num_batches": count,
        "checkpoint": str(cfg.resume_checkpoint),
        "eval/mean_loss_final": sums["eval/loss_final"] / count,
        "eval/mean_loss_traj": sums["eval/loss_traj"] / count,
        "eval/mean_loss_path": sums["eval/loss_path"] / count,
        "eval/mean_loss_macro": sums["eval/loss_macro"] / count,
        "eval/mean_student_teacher_final_mse": sums["eval/student_teacher_final_mse"] / count,
        "eval/mean_student_gt_action_mse": sums["eval/student_gt_action_mse"] / count,
        "eval/mean_teacher_gt_action_mse": sums["eval/teacher_gt_action_mse"] / count,
    }
    _append_jsonl(eval_path, condition)
    _append_jsonl(eval_path, summary)
    print(
        f"eval_summary num_batches={count} "
        f"mean_loss_final={summary['eval/mean_loss_final']:.6f} "
        f"mean_loss_traj={summary['eval/mean_loss_traj']:.6f} "
        f"mean_loss_path={summary['eval/mean_loss_path']:.6f} "
        f"mean_loss_macro={summary['eval/mean_loss_macro']:.6f} "
        f"mean_student_teacher_final_mse={summary['eval/mean_student_teacher_final_mse']:.6f} "
        f"mean_student_gt_action_mse={summary['eval/mean_student_gt_action_mse']:.6f} "
        f"mean_teacher_gt_action_mse={summary['eval/mean_teacher_gt_action_mse']:.6f}",
        flush=True,
    )
    overwatch.info(f"EVAL METRICS SAVED | path={eval_path}")


if __name__ == "__main__":
    try:
        main()
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
