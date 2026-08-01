"""Compare two SimplerEnv rollouts or a GT/teacher/student dataset comparison."""

import argparse
import json
import logging
import textwrap
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


ACTION_FIELDS = {
    "raw": ("raw_actions", "raw_action_dim_names"),
    "executed": ("executed_actions", "executed_action_dim_names"),
}
REQUIRED_METADATA_FIELDS = (
    "model_label",
    "checkpoint",
    "student_action_checkpoint",
    "action_model_type",
    "num_ddim_steps",
    "task_description",
    "env_name",
    "scene_name",
    "control_mode",
    "robot_init_xy",
    "robot_init_quat",
    "object_init_xy",
    "object_episode_id",
    "success",
    "additional_env_build_kwargs_json",
)
PAIR_METADATA_FIELDS = (
    "task_description",
    "env_name",
    "scene_name",
    "control_mode",
    "robot_init_xy",
    "robot_init_quat",
    "object_init_xy",
    "object_episode_id",
    "additional_env_build_kwargs_json",
)
COMPARE_MODE_TEACHER_STUDENT = "teacher-student"
COMPARE_MODE_GT_TEACHER_STUDENT = "gt-teacher-student"
DATASET_REQUIRED_FIELDS = (
    "gt_actions",
    "teacher_actions",
    "student_actions",
    "action_dim_names",
    "data_source",
    "gt_source",
    "action_space",
    "data_mix",
    "data_root_dir",
    "future_action_window_size",
    "action_dim",
    "num_ddim_steps_teacher",
    "num_ddim_steps_student",
    "teacher_checkpoint",
    "student_checkpoint",
    "sample_index",
)
DATASET_OPTIONAL_FIELDS = (
    "prompt_text",
    "task_instruction",
)


def _native_value(value: np.ndarray) -> Any:
    if value.ndim == 0:
        return value.item()
    return value.tolist()


def load_trajectory(path: Path, action_type: str) -> Tuple[np.ndarray, Sequence[str], Dict[str, Any]]:
    action_field, dimension_field = ACTION_FIELDS[action_type]
    with np.load(path, allow_pickle=False) as data:
        required = {action_field, dimension_field, "steps", *REQUIRED_METADATA_FIELDS}
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"{path}: missing required NPZ fields: {', '.join(missing)}")

        actions = np.asarray(data[action_field], dtype=np.float64)
        dimensions = [str(name) for name in data[dimension_field].tolist()]
        steps = np.asarray(data["steps"])
        metadata = {field: _native_value(data[field]) for field in REQUIRED_METADATA_FIELDS}

    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"{path}: {action_field} must have shape [N, 7], got {actions.shape}")
    if actions.shape[0] == 0:
        raise ValueError(f"{path}: {action_field} is empty")
    if not np.isfinite(actions).all():
        raise ValueError(f"{path}: {action_field} contains NaN or Inf")
    if steps.shape != (actions.shape[0],):
        raise ValueError(f"{path}: steps shape {steps.shape} does not match trajectory length {actions.shape[0]}")
    if len(dimensions) != 7:
        raise ValueError(f"{path}: {dimension_field} must contain 7 names, got {len(dimensions)}")
    return actions, dimensions, metadata


def load_dataset_comparison(path: Path) -> Tuple[Dict[str, np.ndarray], Sequence[str], Dict[str, Any]]:
    with np.load(path, allow_pickle=False) as data:
        missing = sorted(set(DATASET_REQUIRED_FIELDS).difference(data.files))
        if missing:
            raise ValueError(f"{path}: missing required dataset comparison NPZ fields: {', '.join(missing)}")
        trajectories = {
            name: np.asarray(data[name], dtype=np.float64)
            for name in ("gt_actions", "teacher_actions", "student_actions")
        }
        dimensions = [str(name) for name in data["action_dim_names"].tolist()]
        metadata = {field: _native_value(data[field]) for field in DATASET_REQUIRED_FIELDS[4:]}
        for field in DATASET_OPTIONAL_FIELDS:
            if field in data.files:
                metadata[field] = _native_value(data[field])

    expected_shape = trajectories["gt_actions"].shape
    if len(expected_shape) != 2 or expected_shape[1] != 7 or expected_shape[0] == 0:
        raise ValueError(f"{path}: gt_actions must have non-empty shape [T, 7], got {expected_shape}")
    if int(metadata["action_dim"]) != expected_shape[1]:
        raise ValueError(
            f"{path}: action_dim metadata is {metadata['action_dim']!r}, expected {expected_shape[1]}"
        )
    if len(dimensions) != expected_shape[1]:
        raise ValueError(
            f"{path}: action_dim_names must contain {expected_shape[1]} names, got {len(dimensions)}"
        )

    for name, actions in trajectories.items():
        if actions.shape != expected_shape:
            raise ValueError(
                f"{path}: {name} shape {actions.shape} does not match gt_actions shape {expected_shape}"
            )
        if not np.isfinite(actions).all():
            raise ValueError(f"{path}: {name} contains NaN or Inf")
    return trajectories, dimensions, metadata


def validate_pair(
    teacher_dimensions: Sequence[str],
    student_dimensions: Sequence[str],
    teacher_metadata: Dict[str, Any],
    student_metadata: Dict[str, Any],
) -> Sequence[str]:
    mismatches = []
    if list(teacher_dimensions) != list(student_dimensions):
        mismatches.append(
            f"action dimension names: teacher={list(teacher_dimensions)!r}, student={list(student_dimensions)!r}"
        )

    teacher_label = str(teacher_metadata["model_label"]).lower()
    student_label = str(student_metadata["model_label"]).lower()
    if teacher_label != "teacher":
        mismatches.append(f"teacher model_label is {teacher_metadata['model_label']!r}, expected 'teacher'")
    if student_label != "student":
        mismatches.append(f"student model_label is {student_metadata['model_label']!r}, expected 'student'")

    for field in PAIR_METADATA_FIELDS:
        teacher_value = teacher_metadata[field]
        student_value = student_metadata[field]
        if not _metadata_equal(teacher_value, student_value):
            mismatches.append(f"{field}: teacher={teacher_value!r}, student={student_value!r}")
    return mismatches


def _metadata_equal(left: Any, right: Any) -> bool:
    if isinstance(left, list) or isinstance(right, list):
        try:
            return bool(np.allclose(np.asarray(left), np.asarray(right), rtol=0.0, atol=1e-8))
        except (TypeError, ValueError):
            return left == right
    return left == right


def compute_common_prefix_metrics(teacher: np.ndarray, student: np.ndarray) -> Dict[str, Any]:
    common_length = min(len(teacher), len(student))
    difference = teacher[:common_length] - student[:common_length]
    return {
        "comparison_scope": "common closed-loop simulator-step prefix",
        "common_length": common_length,
        "teacher_length": len(teacher),
        "student_length": len(student),
        "mse_overall": float(np.mean(np.square(difference))),
        "mae_overall": float(np.mean(np.abs(difference))),
        "mse_per_dimension": np.mean(np.square(difference), axis=0).tolist(),
        "mae_per_dimension": np.mean(np.abs(difference), axis=0).tolist(),
    }


def compute_dataset_triplet_metrics(
    gt: np.ndarray,
    teacher: np.ndarray,
    student: np.ndarray,
) -> Dict[str, Any]:
    if gt.shape != teacher.shape or teacher.shape != student.shape:
        raise ValueError(
            f"dataset trajectories must share one shape, got gt={gt.shape}, teacher={teacher.shape}, "
            f"student={student.shape}"
        )

    def pair_metrics(left: np.ndarray, right: np.ndarray) -> Tuple[float, float, list[float], list[float]]:
        difference = left - right
        return (
            float(np.mean(np.square(difference))),
            float(np.mean(np.abs(difference))),
            np.mean(np.square(difference), axis=0).tolist(),
            np.mean(np.abs(difference), axis=0).tolist(),
        )

    student_gt_mse, student_gt_mae, student_gt_mse_dim, student_gt_mae_dim = pair_metrics(student, gt)
    teacher_gt_mse, teacher_gt_mae, teacher_gt_mse_dim, teacher_gt_mae_dim = pair_metrics(teacher, gt)
    student_teacher_mse, student_teacher_mae, student_teacher_mse_dim, student_teacher_mae_dim = pair_metrics(
        student, teacher
    )
    return {
        "comparison_scope": "same training-dataloader sample action horizon",
        "trajectory_length": int(gt.shape[0]),
        "student_gt_mse": student_gt_mse,
        "student_gt_mae": student_gt_mae,
        "student_gt_mse_per_dimension": student_gt_mse_dim,
        "student_gt_mae_per_dimension": student_gt_mae_dim,
        "teacher_gt_mse": teacher_gt_mse,
        "teacher_gt_mae": teacher_gt_mae,
        "teacher_gt_mse_per_dimension": teacher_gt_mse_dim,
        "teacher_gt_mae_per_dimension": teacher_gt_mae_dim,
        "student_teacher_mse": student_teacher_mse,
        "student_teacher_mae": student_teacher_mae,
        "student_teacher_mse_per_dimension": student_teacher_mse_dim,
        "student_teacher_mae_per_dimension": student_teacher_mae_dim,
    }


def plot_teacher_vs_student(
    teacher: np.ndarray,
    student: np.ndarray,
    dimension_names: Sequence[str],
    teacher_metadata: Dict[str, Any],
    student_metadata: Dict[str, Any],
    action_type: str,
    save_path: Path,
    pair_verified: bool,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(7, 1, figsize=(12, 15), sharex=True)
    teacher_steps = np.arange(len(teacher))
    student_steps = np.arange(len(student))

    for index, axis in enumerate(axes):
        axis.plot(teacher_steps, teacher[:, index], label="Teacher", color="#2166ac", linewidth=1.6)
        axis.plot(
            student_steps,
            student[:, index],
            label="Student",
            color="#b2182b",
            linewidth=1.4,
            linestyle="--",
        )
        axis.set_ylabel(dimension_names[index])
        axis.grid(True, alpha=0.3)
        axis.legend(loc="upper right")

    axes[-1].set_xlabel("Simulator Step")
    verification = "verified pair" if pair_verified else "UNVERIFIED PAIR"
    teacher_checkpoint = Path(str(teacher_metadata["checkpoint"])).name
    student_checkpoint = Path(
        str(student_metadata["student_action_checkpoint"] or student_metadata["checkpoint"])
    ).name
    title = (
        f"Teacher vs Student {action_type.title()} Actions | {verification}\n"
        f"{teacher_metadata['scene_name']} | {teacher_metadata['env_name']}\n"
        f"Teacher: {teacher_checkpoint}, {len(teacher)} steps, DDIM={teacher_metadata['num_ddim_steps']}, "
        f"success={teacher_metadata['success']}\n"
        f"Student: {student_checkpoint}, {len(student)} steps, "
        f"DDIM={student_metadata['num_ddim_steps']}, success={student_metadata['success']}"
    )
    figure.suptitle(title, fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    figure.savefig(save_path, dpi=300)
    plt.close(figure)


def plot_gt_teacher_student(
    gt: np.ndarray,
    teacher: np.ndarray,
    student: np.ndarray,
    dimension_names: Sequence[str],
    metadata: Dict[str, Any],
    save_path: Path,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(len(dimension_names), 1, figsize=(12, 15), sharex=True)
    steps = np.arange(len(gt))

    for index, axis in enumerate(axes):
        axis.plot(steps, gt[:, index], label="GT", color="#222222", linewidth=1.6)
        axis.plot(steps, teacher[:, index], label="Teacher", color="#2166ac", linewidth=1.5)
        axis.plot(
            steps,
            student[:, index],
            label="Student",
            color="#b2182b",
            linewidth=1.4,
            linestyle="--",
        )
        axis.set_ylabel(dimension_names[index])
        axis.grid(True, alpha=0.3)
        axis.legend(loc="upper right")

    teacher_checkpoint = Path(str(metadata["teacher_checkpoint"])).name
    student_checkpoint = Path(str(metadata["student_checkpoint"])).name
    instruction = str(metadata.get("task_instruction") or metadata.get("prompt_text") or "").strip()
    instruction_line = ""
    if instruction:
        instruction_line = f"Instruction: {textwrap.shorten(instruction, width=130, placeholder='...')}\n"
    title = (
        "Training-Dataloader GT vs Teacher vs Student\n"
        f"{instruction_line}"
        f"sample={metadata['sample_index']} | data_mix={metadata['data_mix']} | "
        f"action_space={metadata['action_space']}\n"
        f"Teacher: {teacher_checkpoint}, DDIM={metadata['num_ddim_steps_teacher']} | "
        f"Student: {student_checkpoint}, DDIM={metadata['num_ddim_steps_student']}"
    )
    axes[-1].set_xlabel("Future Action Step")
    figure.suptitle(title, fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    figure.savefig(save_path, dpi=300)
    plt.close(figure)


def _checkpoint_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "model_label": metadata["model_label"],
        "checkpoint": metadata["checkpoint"],
        "student_action_checkpoint": metadata["student_action_checkpoint"],
        "action_model_type": metadata["action_model_type"],
        "num_ddim_steps": metadata["num_ddim_steps"],
        "success": metadata["success"],
    }


def _compare_teacher_student(args: argparse.Namespace) -> Path:
    teacher, teacher_dimensions, teacher_metadata = load_trajectory(args.teacher_npz, args.action_type)
    student, student_dimensions, student_metadata = load_trajectory(args.student_npz, args.action_type)
    mismatches = validate_pair(teacher_dimensions, student_dimensions, teacher_metadata, student_metadata)
    if mismatches and not args.allow_metadata_mismatch:
        details = "\n  - ".join(mismatches)
        raise ValueError(f"Teacher/Student NPZ metadata mismatch:\n  - {details}")
    for mismatch in mismatches:
        logging.warning("Unverified pair: %s", mismatch)

    metrics = compute_common_prefix_metrics(teacher, student)
    metrics.update(
        {
            "action_type": args.action_type,
            "dimension_names": list(teacher_dimensions),
            "pair_verified": not mismatches,
            "metadata_mismatches": list(mismatches),
            "teacher": _checkpoint_metadata(teacher_metadata),
            "student": _checkpoint_metadata(student_metadata),
        }
    )
    plot_teacher_vs_student(
        teacher,
        student,
        teacher_dimensions,
        teacher_metadata,
        student_metadata,
        args.action_type,
        args.save_path,
        pair_verified=not mismatches,
    )

    metrics_path = args.metrics_path or args.save_path.with_name(f"{args.save_path.stem}_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    logging.info("Comparison plot saved to %s", args.save_path)
    logging.info("Comparison metrics saved to %s", metrics_path)
    return metrics_path


def _compare_gt_teacher_student(args: argparse.Namespace) -> Path:
    trajectories, dimension_names, metadata = load_dataset_comparison(args.dataset_npz)
    metrics = compute_dataset_triplet_metrics(
        trajectories["gt_actions"],
        trajectories["teacher_actions"],
        trajectories["student_actions"],
    )
    metrics.update(
        {
            "compare_mode": COMPARE_MODE_GT_TEACHER_STUDENT,
            "dimension_names": list(dimension_names),
            "dataset": {
                "data_source": metadata["data_source"],
                "gt_source": metadata["gt_source"],
                "action_space": metadata["action_space"],
                "data_mix": metadata["data_mix"],
                "data_root_dir": metadata["data_root_dir"],
                "future_action_window_size": metadata["future_action_window_size"],
                "sample_index": metadata["sample_index"],
                "task_instruction": metadata.get("task_instruction", ""),
                "prompt_text": metadata.get("prompt_text", ""),
            },
            "teacher": {
                "checkpoint": metadata["teacher_checkpoint"],
                "num_ddim_steps": metadata["num_ddim_steps_teacher"],
            },
            "student": {
                "checkpoint": metadata["student_checkpoint"],
                "num_ddim_steps": metadata["num_ddim_steps_student"],
            },
        }
    )
    plot_gt_teacher_student(
        trajectories["gt_actions"],
        trajectories["teacher_actions"],
        trajectories["student_actions"],
        dimension_names,
        metadata,
        args.save_path,
    )

    metrics_path = args.metrics_path or args.save_path.with_name(f"{args.save_path.stem}_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    logging.info("Comparison plot saved to %s", args.save_path)
    logging.info("Comparison metrics saved to %s", metrics_path)
    return metrics_path


def compare(args: argparse.Namespace) -> Path:
    compare_mode = getattr(args, "compare_mode", COMPARE_MODE_TEACHER_STUDENT)
    if compare_mode == COMPARE_MODE_TEACHER_STUDENT:
        return _compare_teacher_student(args)
    if compare_mode == COMPARE_MODE_GT_TEACHER_STUDENT:
        return _compare_gt_teacher_student(args)
    raise ValueError(f"Unsupported compare_mode: {compare_mode!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare SimplerEnv or training-dataloader action trajectories")
    parser.add_argument(
        "--compare-mode",
        choices=(COMPARE_MODE_TEACHER_STUDENT, COMPARE_MODE_GT_TEACHER_STUDENT),
        default=COMPARE_MODE_TEACHER_STUDENT,
    )
    parser.add_argument("--teacher-npz", type=Path)
    parser.add_argument("--student-npz", type=Path)
    parser.add_argument("--dataset-npz", type=Path)
    parser.add_argument("--action-type", choices=sorted(ACTION_FIELDS), default="raw")
    parser.add_argument("--save-path", type=Path, required=True)
    parser.add_argument("--metrics-path", type=Path, default=None)
    parser.add_argument(
        "--allow-metadata-mismatch",
        action="store_true",
        help="Draw an explicitly marked unverified comparison when rollout metadata differs",
    )
    args = parser.parse_args()
    if args.compare_mode == COMPARE_MODE_TEACHER_STUDENT:
        if args.teacher_npz is None or args.student_npz is None:
            parser.error("--compare-mode teacher-student requires --teacher-npz and --student-npz")
        if args.dataset_npz is not None:
            parser.error("--dataset-npz is only valid with --compare-mode gt-teacher-student")
    else:
        if args.dataset_npz is None:
            parser.error("--compare-mode gt-teacher-student requires --dataset-npz")
        if args.teacher_npz is not None or args.student_npz is not None:
            parser.error("--teacher-npz and --student-npz are only valid with --compare-mode teacher-student")
        if args.allow_metadata_mismatch:
            parser.error("--allow-metadata-mismatch is only valid with --compare-mode teacher-student")
    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    compare(parse_args())


if __name__ == "__main__":
    main()
