"""
Export GT / teacher / student action-horizon comparisons from the training dataloader.

Example:
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python scripts/export_train_compare_npz.py \
      --data_root_dir "/home/huangjiaqi/.vscode-server/openvla_data" \
      --pretrained_checkpoint "/data/huangjiaqi/projects/CogACT-Base" \
      --resume_checkpoint runs/distillation/checkpoint_final.pt \
      --data_mix bridge \
      --batch_size 1 \
      --max_batches 2 \
      --num_ddim_steps_teacher 20 \
      --num_ddim_steps_student 4
"""

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

import draccus
import numpy as np
import torch

from action_model.action_model import ActionModel
from conf import DistillationConfig
from distillation.checkpoint import load_checkpoint
from distillation.loaders import load_dataloader, load_student, load_teacher
from distillation.runners import (
    get_student_timesteps,
    run_student_ddim_with_recording,
    run_teacher_with_recording,
)


ACTION_DIM_NAMES = np.array(
    [
        "world_vector_x",
        "world_vector_y",
        "world_vector_z",
        "rotation_delta_roll",
        "rotation_delta_pitch",
        "rotation_delta_yaw",
        "open_gripper",
    ]
)


@dataclass
class ExportTrainCompareConfig(DistillationConfig):
    export_dir: Path = Path("runs/distillation/train_compare_npz")
    export_prefix: str = "train_compare"
    max_batches: Optional[int] = 1
    max_samples: Optional[int] = None


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


def _move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        key: value.to(device)
        if isinstance(value, torch.Tensor)
        else {inner_key: inner_value.to(device) for inner_key, inner_value in value.items()}
        if isinstance(value, dict)
        else value
        for key, value in batch.items()
    }


def _actions_future(batch: Dict[str, Any], cfg: DistillationConfig) -> torch.Tensor:
    batch_size = batch["actions"].shape[0]
    horizon = cfg.future_action_window_size + 1
    expected_shape = (batch_size, horizon, cfg.action_dim)
    actions = batch["actions"][:, -horizon:, :]
    if tuple(actions.shape) != expected_shape:
        raise RuntimeError(f"actions_future shape mismatch: got={tuple(actions.shape)}, expected={expected_shape}")
    return actions


def _finite_numpy(tensor: torch.Tensor, name: str) -> np.ndarray:
    array = tensor.detach().float().cpu().numpy()
    if not np.isfinite(array).all():
        raise RuntimeError(f"{name} contains NaN or Inf")
    return array


def _decode_prompt(tokenizer: Any, input_ids: Sequence[int]) -> str:
    if tokenizer is None:
        return ""
    return str(tokenizer.decode(list(input_ids), skip_special_tokens=True)).strip()


def _extract_task_instruction(prompt_text: str) -> str:
    text = prompt_text.strip()
    if text.startswith("In:"):
        text = text[len("In:") :].strip()
    if "\nOut:" in text:
        text = text.split("\nOut:", 1)[0].strip()
    return text


def _run_compare_batch(
    *,
    teacher,
    student: ActionModel,
    batch: Dict[str, Any],
    cfg: DistillationConfig,
    device: torch.device,
) -> Dict[str, Any]:
    actions_future = _actions_future(batch, cfg)
    noise = torch.randn_like(actions_future)
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
    if x0_teacher.shape != x0_student.shape or x0_student.shape != actions_future.shape:
        raise RuntimeError(
            "compare output shape mismatch: "
            f"gt={tuple(actions_future.shape)}, "
            f"teacher={tuple(x0_teacher.shape)}, "
            f"student={tuple(x0_student.shape)}"
        )

    return {
        "gt_actions": actions_future,
        "teacher_actions": x0_teacher,
        "student_actions": x0_student,
        "noise": noise,
        "teacher_timesteps": teacher_out["timesteps"],
        "student_timesteps": student_out["timesteps"],
        "teacher_full_timesteps": teacher_out["teacher_full_timesteps"],
    }


def _metadata_row(
    *,
    cfg: ExportTrainCompareConfig,
    teacher_checkpoint: Path,
    batch_index: int,
    batch_sample_index: int,
    sample_index: int,
    npz_path: Path,
) -> Dict[str, Any]:
    return {
        "path": str(npz_path),
        "data_source": "training_dataloader",
        "gt_source": 'batch["actions"][:, -(future_action_window_size + 1):, :]',
        "action_space": "normalized",
        "data_mix": cfg.data_mix,
        "data_root_dir": str(cfg.data_root_dir),
        "future_action_window_size": cfg.future_action_window_size,
        "past_action_window_size": cfg.past_action_window_size,
        "action_dim": cfg.action_dim,
        "num_ddim_steps_teacher": cfg.num_ddim_steps_teacher,
        "num_ddim_steps_student": cfg.num_ddim_steps_student,
        "teacher_checkpoint": str(teacher_checkpoint),
        "student_checkpoint": str(cfg.resume_checkpoint),
        "sample_index": sample_index,
        "batch_index": batch_index,
        "batch_sample_index": batch_sample_index,
        "eval_seed": cfg.eval_seed,
    }


def _save_sample_npz(
    *,
    path: Path,
    outputs: Dict[str, Any],
    batch: Dict[str, Any],
    metadata: Dict[str, Any],
    sample_in_batch: int,
    tokenizer: Any,
) -> None:
    payload = {
        "gt_actions": _finite_numpy(outputs["gt_actions"][sample_in_batch], "gt_actions"),
        "teacher_actions": _finite_numpy(outputs["teacher_actions"][sample_in_batch], "teacher_actions"),
        "student_actions": _finite_numpy(outputs["student_actions"][sample_in_batch], "student_actions"),
        "noise": _finite_numpy(outputs["noise"][sample_in_batch], "noise"),
        "action_dim_names": ACTION_DIM_NAMES,
        "teacher_timesteps": np.asarray(outputs["teacher_timesteps"], dtype=np.int64),
        "student_timesteps": np.asarray(outputs["student_timesteps"], dtype=np.int64),
        "teacher_full_timesteps": np.asarray(outputs["teacher_full_timesteps"], dtype=np.int64),
    }
    if "input_ids" in batch and isinstance(batch["input_ids"], torch.Tensor):
        input_ids = batch["input_ids"][sample_in_batch].detach().cpu().numpy()
        prompt_text = _decode_prompt(tokenizer, input_ids.tolist())
        payload["input_ids"] = input_ids
        payload["prompt_text"] = np.array(prompt_text)
        payload["task_instruction"] = np.array(_extract_task_instruction(prompt_text))
    if "attention_mask" in batch and isinstance(batch["attention_mask"], torch.Tensor):
        payload["attention_mask"] = batch["attention_mask"][sample_in_batch].detach().cpu().numpy()

    for key, value in metadata.items():
        payload[key] = np.array(value)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


@draccus.wrap()
def main(cfg: ExportTrainCompareConfig) -> None:
    if cfg.resume_checkpoint is None:
        raise ValueError("export requires --resume_checkpoint pointing to a distilled student checkpoint")
    if cfg.max_batches is None and cfg.max_samples is None:
        raise ValueError("export requires --max_batches or --max_samples to avoid unbounded training-data export")

    torch.manual_seed(cfg.eval_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.eval_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    teacher_checkpoint = _resolve_teacher_checkpoint(cfg)
    teacher = load_teacher(
        teacher_checkpoint,
        cfg.action_model_type_teacher,
        cfg.future_action_window_size,
        hf_token=os.environ.get("HF_TOKEN"),
    )
    teacher = teacher.to(device)
    teacher.eval()
    tokenizer = teacher.llm_backbone.get_tokenizer()

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
    student.eval()

    dataloader = load_dataloader(
        teacher,
        cfg.data_root_dir,
        cfg.data_mix,
        cfg.batch_size,
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        train=True,
        image_aug=False,
        load_all_data_for_training=cfg.load_all_data_for_training,
        future_action_window_size=cfg.future_action_window_size,
        past_action_window_size=cfg.past_action_window_size,
    )

    cfg.export_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cfg.export_dir / f"{cfg.export_prefix}_manifest.jsonl"
    manifest_path.write_text("", encoding="utf-8")

    exported = 0
    for batch_index, batch in enumerate(dataloader, start=1):
        if cfg.max_batches is not None and batch_index > cfg.max_batches:
            break
        batch = _move_batch_to_device(batch, device)
        outputs = _run_compare_batch(
            teacher=teacher,
            student=student,
            batch=batch,
            cfg=cfg,
            device=device,
        )

        batch_size = int(outputs["gt_actions"].shape[0])
        for batch_sample_index in range(batch_size):
            if cfg.max_samples is not None and exported >= cfg.max_samples:
                break
            sample_index = exported
            npz_path = cfg.export_dir / f"{cfg.export_prefix}_{sample_index:06d}.npz"
            metadata = _metadata_row(
                cfg=cfg,
                teacher_checkpoint=teacher_checkpoint,
                batch_index=batch_index,
                batch_sample_index=batch_sample_index,
                sample_index=sample_index,
                npz_path=npz_path,
            )
            _save_sample_npz(
                path=npz_path,
                outputs=outputs,
                batch=batch,
                metadata=metadata,
                sample_in_batch=batch_sample_index,
                tokenizer=tokenizer,
            )
            with manifest_path.open("a", encoding="utf-8") as manifest:
                manifest.write(json.dumps(metadata, ensure_ascii=False) + "\n")
            exported += 1
            print(f"exported sample_index={sample_index} path={npz_path}", flush=True)

        if cfg.max_samples is not None and exported >= cfg.max_samples:
            break

    if exported == 0:
        raise RuntimeError("export produced zero samples")
    print(f"export_summary samples={exported} manifest={manifest_path}", flush=True)


if __name__ == "__main__":
    main()
