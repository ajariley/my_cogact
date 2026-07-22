import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist

from action_model.action_model import ActionModel
from conf.distillation import DistillationConfig
from distillation.loaders import load_dataloader
from distillation.path import depth_path_losses
from distillation.runners import (
    get_student_timesteps,
    run_student_ddim_with_recording,
    run_teacher_with_recording,
)


def _move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else
        {kk: vv.to(device) for kk, vv in v.items()} if isinstance(v, dict) else v
        for k, v in batch.items()
    }


def _batch_to_cpu(batch: Dict[str, Any]) -> Dict[str, Any]:
    return {
        k: v.detach().cpu() if isinstance(v, torch.Tensor) else
        {kk: vv.detach().cpu() for kk, vv in v.items()} if isinstance(v, dict) else v
        for k, v in batch.items()
    }


def _actions_future(batch: Dict[str, Any], cfg: DistillationConfig) -> torch.Tensor:
    batch_size = batch["actions"].shape[0]
    horizon = cfg.future_action_window_size + 1
    expected_shape = (batch_size, horizon, cfg.action_dim)
    actions = batch["actions"][:, -horizon:, :]
    if tuple(actions.shape) != expected_shape:
        raise RuntimeError(f"actions_future shape mismatch: got={tuple(actions.shape)}, expected={expected_shape}")
    return actions


def _load_or_build_train_eval_cache(
    *,
    teacher,
    cfg: DistillationConfig,
    device: torch.device,
    max_eval_batches: int,
) -> tuple[Dict[str, Any], Path]:
    cache_path = cfg.eval_cache_path or (
        cfg.output_dir / f"train_eval_cache_seed{cfg.eval_seed}_b{cfg.batch_size}_n{max_eval_batches}.pt"
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    is_dist = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_dist else 0
    if cache_path.exists():
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        if len(cache.get("items", [])) == max_eval_batches:
            return cache, cache_path
        if rank == 0:
            cache_path.unlink()

    if is_dist and rank != 0:
        dist.barrier()
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        if len(cache.get("items", [])) != max_eval_batches:
            raise RuntimeError(
                f"invalid train eval cache at {cache_path}: "
                f"cached_batches={len(cache.get('items', []))}, requested_batches={max_eval_batches}"
            )
        return cache, cache_path

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
        disable_distributed_shard=True,
    )

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
        raise RuntimeError("train eval cache produced zero batches")

    cache = {
        "eval_seed": cfg.eval_seed,
        "batch_size": cfg.batch_size,
        "max_batches": max_eval_batches,
        "future_action_window_size": cfg.future_action_window_size,
        "action_dim": cfg.action_dim,
        "items": items,
    }
    torch.save(cache, cache_path)
    if is_dist:
        dist.barrier()
    return cache, cache_path


def run_train_eval(
    *,
    teacher,
    student: ActionModel,
    cfg: DistillationConfig,
    device: torch.device,
    max_eval_batches: int,
) -> Dict[str, Any]:
    was_training = student.training
    student.eval()
    eval_cache, eval_cache_path = _load_or_build_train_eval_cache(
        teacher=teacher,
        cfg=cfg,
        device=device,
        max_eval_batches=max_eval_batches,
    )

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
    torch.manual_seed(cfg.eval_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.eval_seed)

    with torch.no_grad():
        for item in eval_cache["items"]:
            count += 1
            batch = _move_batch_to_device(item["batch"], device)
            actions_future = _actions_future(batch, cfg)
            noise = item["noise"].to(device=device, dtype=actions_future.dtype)
            student_timesteps = get_student_timesteps(student, cfg.num_ddim_steps_student)
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
                    "train eval output shape mismatch: "
                    f"actions_future={tuple(actions_future.shape)}, "
                    f"x0_teacher={tuple(x0_teacher.shape)}, "
                    f"x0_student={tuple(x0_student.shape)}"
                )
            if teacher_traj.shape != student_traj.shape:
                raise RuntimeError(
                    "train eval trajectory shape mismatch: "
                    f"teacher={tuple(teacher_traj.shape)}, student={tuple(student_traj.shape)}"
                )
            loss_final = float(torch.nn.functional.mse_loss(x0_student, x0_teacher).item())
            loss_traj = float(torch.nn.functional.mse_loss(student_traj, teacher_traj).item())
            loss_path_tensor, loss_macro_tensor, _ = depth_path_losses(
                student_out["depth_x0_path"],
                teacher_out["depth_x0_path"],
                action_dim_weights=cfg.refinement_action_dim_weights,
                eps=cfg.refinement_progress_eps,
            )
            loss_path = float(loss_path_tensor.item())
            loss_macro = float(loss_macro_tensor.item())
            sums["eval/loss_final"] += loss_final
            sums["eval/loss_traj"] += loss_traj
            sums["eval/loss_path"] += loss_path
            sums["eval/loss_macro"] += loss_macro
            sums["eval/student_teacher_final_mse"] += loss_final
            sums["eval/student_gt_action_mse"] += float(torch.nn.functional.mse_loss(x0_student, actions_future).item())
            sums["eval/teacher_gt_action_mse"] += float(torch.nn.functional.mse_loss(x0_teacher, actions_future).item())

    if count == 0:
        raise RuntimeError("train eval produced zero batches")
    if was_training:
        student.train()

    local = torch.tensor(
        [
            sums["eval/loss_final"],
            sums["eval/loss_traj"],
            sums["eval/loss_path"],
            sums["eval/loss_macro"],
            sums["eval/student_teacher_final_mse"],
            sums["eval/student_gt_action_mse"],
            sums["eval/teacher_gt_action_mse"],
            float(count),
        ],
        device=device,
        dtype=torch.float64,
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(local, op=dist.ReduceOp.SUM)

    total_count = int(local[7].item())
    return {
        "type": "train_eval_summary",
        "eval/num_batches": total_count,
        "eval_cache_path": str(eval_cache_path),
        "eval/mean_loss_final": float(local[0].item() / total_count),
        "eval/mean_loss_traj": float(local[1].item() / total_count),
        "eval/mean_loss_path": float(local[2].item() / total_count),
        "eval/mean_loss_macro": float(local[3].item() / total_count),
        "eval/mean_student_teacher_final_mse": float(local[4].item() / total_count),
        "eval/mean_student_gt_action_mse": float(local[5].item() / total_count),
        "eval/mean_teacher_gt_action_mse": float(local[6].item() / total_count),
    }


class TrainController:
    """
    控制训练过程中的early stopping。
    """

    def __init__(
        self,
        *,
        eval_interval: int,
        patience: int,
        min_delta: float,
        metric_name: str,
        output_dir: Path,
        checkpoint_suffix: str = "",
        enabled: bool = True,
    ) -> None:
        self.eval_interval = eval_interval
        self.patience = patience
        self.min_delta = min_delta
        self.metric_name = metric_name
        self.output_dir = output_dir
        self.checkpoint_suffix = checkpoint_suffix
        self.enabled = enabled
        self.best_metric: Optional[float] = None
        self.best_checkpoint_path: Optional[Path] = None
        self.bad_eval_count = 0

    def should_eval(self, step: int) -> bool:
        return self.enabled and self.eval_interval > 0 and step > 0 and step % self.eval_interval == 0

    def update(self, *, step: int, checkpoint_path: Path, eval_summary: Dict[str, Any]) -> bool:
        if self.metric_name not in eval_summary:
            raise KeyError(f"early stop metric `{self.metric_name}` not found in eval summary")
        metric = float(eval_summary[self.metric_name])
        improved = self.best_metric is None or metric < self.best_metric - self.min_delta
        if improved:
            self.best_metric = metric
            self.best_checkpoint_path = checkpoint_path
            self.bad_eval_count = 0
            self._copy_best_checkpoint(checkpoint_path)
        else:
            self.bad_eval_count += 1
        self._append_state(step=step, checkpoint_path=checkpoint_path, eval_summary=eval_summary, improved=improved)
        return self.bad_eval_count >= self.patience

    def best_checkpoint_alias(self) -> Path:
        suffix_part = f"_{self.checkpoint_suffix}" if self.checkpoint_suffix else ""
        return self.output_dir / f"checkpoint_best{suffix_part}.pt"

    def _copy_best_checkpoint(self, checkpoint_path: Path) -> None:
        alias = self.best_checkpoint_alias()
        alias.parent.mkdir(parents=True, exist_ok=True)
        if checkpoint_path.resolve() != alias.resolve():
            shutil.copy2(checkpoint_path, alias)

    def _append_state(
        self,
        *,
        step: int,
        checkpoint_path: Path,
        eval_summary: Dict[str, Any],
        improved: bool,
    ) -> None:
        row = {
            "type": "train_controller",
            "step": step,
            "checkpoint": str(checkpoint_path),
            "metric_name": self.metric_name,
            "metric": float(eval_summary[self.metric_name]),
            "best_metric": self.best_metric,
            "best_checkpoint": str(self.best_checkpoint_path) if self.best_checkpoint_path else None,
            "bad_eval_count": self.bad_eval_count,
            "patience": self.patience,
            "improved": improved,
            **eval_summary,
        }
        path = self.output_dir / "train_controller.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
