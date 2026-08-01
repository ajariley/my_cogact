import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from draw_compare.experiment.eval_policy import compare, compute_common_prefix_metrics, load_trajectory


DIMENSIONS = np.array(
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


def _write_npz(path: Path, actions: np.ndarray, label: str, *, scene: str = "scene") -> None:
    np.savez_compressed(
        path,
        raw_actions=actions,
        executed_actions=actions,
        steps=np.arange(len(actions)),
        raw_action_dim_names=DIMENSIONS,
        executed_action_dim_names=DIMENSIONS,
        model_label=np.array(label),
        checkpoint=np.array(f"{label}.pt"),
        student_action_checkpoint=np.array("student.pt" if label == "student" else ""),
        action_model_type=np.array("DiT-B"),
        num_ddim_steps=np.array(20 if label == "teacher" else 4),
        task_description=np.array("pick object"),
        env_name=np.array("Env-v0"),
        scene_name=np.array(scene),
        control_mode=np.array("control"),
        robot_init_xy=np.array([0.35, 0.2]),
        robot_init_quat=np.array([1.0, 0.0, 0.0, 0.0]),
        object_init_xy=np.array([-0.1, 0.2]),
        object_episode_id=np.array(-1),
        success=np.array("success"),
        additional_env_build_kwargs_json=np.array('{"flag": true}'),
    )


def test_metrics_use_common_prefix_for_different_lengths():
    teacher = np.zeros((3, 7))
    student = np.ones((2, 7))
    metrics = compute_common_prefix_metrics(teacher, student)
    assert metrics["common_length"] == 2
    assert metrics["teacher_length"] == 3
    assert metrics["student_length"] == 2
    assert metrics["mse_overall"] == pytest.approx(1.0)
    assert metrics["mae_overall"] == pytest.approx(1.0)


def test_compare_writes_plot_and_metrics_for_different_lengths(tmp_path):
    teacher_path = tmp_path / "teacher.npz"
    student_path = tmp_path / "student.npz"
    plot_path = tmp_path / "compare.png"
    _write_npz(teacher_path, np.zeros((3, 7)), "teacher")
    _write_npz(student_path, np.ones((2, 7)), "student")

    metrics_path = compare(
        argparse.Namespace(
            teacher_npz=teacher_path,
            student_npz=student_path,
            action_type="raw",
            save_path=plot_path,
            metrics_path=None,
            allow_metadata_mismatch=False,
        )
    )

    assert plot_path.stat().st_size > 0
    metrics = json.loads(metrics_path.read_text())
    assert metrics["pair_verified"] is True
    assert metrics["common_length"] == 2
    assert metrics["mse_overall"] == pytest.approx(1.0)


def test_compare_rejects_metadata_mismatch(tmp_path):
    teacher_path = tmp_path / "teacher.npz"
    student_path = tmp_path / "student.npz"
    _write_npz(teacher_path, np.zeros((2, 7)), "teacher", scene="teacher-scene")
    _write_npz(student_path, np.zeros((2, 7)), "student", scene="student-scene")

    with pytest.raises(ValueError, match="scene_name"):
        compare(
            argparse.Namespace(
                teacher_npz=teacher_path,
                student_npz=student_path,
                action_type="raw",
                save_path=tmp_path / "compare.png",
                metrics_path=None,
                allow_metadata_mismatch=False,
            )
        )


@pytest.mark.parametrize("actions", [np.zeros((2, 6)), np.full((2, 7), np.nan)])
def test_load_trajectory_rejects_invalid_actions(tmp_path, actions):
    path = tmp_path / "invalid.npz"
    _write_npz(path, actions, "teacher")
    with pytest.raises(ValueError):
        load_trajectory(path, "raw")
