import unittest

import torch

from conf.distillation import DistillationConfig
from distillation.loss import compute_loss
from distillation.path import (
    compress_teacher_path,
    compute_refinement_progress,
    depth_path_losses,
)


def scalar_path(values, batch_size=1):
    return torch.tensor(values, dtype=torch.float32).view(-1, 1, 1, 1).expand(-1, batch_size, 1, 1).clone()


class RefinementProgressTest(unittest.TestCase):
    def test_progress_follows_action_distance(self):
        path = scalar_path([0.0, 1.0, 3.0])

        progress = compute_refinement_progress(path)

        expected = torch.tensor([[0.0], [1.0 / 3.0], [1.0]])
        torch.testing.assert_close(progress, expected)

    def test_progress_is_computed_per_batch_sample(self):
        path = torch.tensor(
            [
                [[[0.0]], [[0.0]]],
                [[[1.0]], [[3.0]]],
                [[[3.0]], [[4.0]]],
            ]
        )

        progress = compute_refinement_progress(path)

        expected = torch.tensor(
            [
                [0.0, 0.0],
                [1.0 / 3.0, 3.0 / 4.0],
                [1.0, 1.0],
            ]
        )
        torch.testing.assert_close(progress, expected)

    def test_action_dimension_weights_change_progress(self):
        path = torch.tensor(
            [
                [[[0.0, 0.0]]],
                [[[1.0, 0.0]]],
                [[[1.0, 1.0]]],
            ]
        )

        progress = compute_refinement_progress(
            path,
            action_dim_weights=torch.tensor([4.0, 1.0]),
        )

        torch.testing.assert_close(progress[:, 0], torch.tensor([0.0, 2.0 / 3.0, 1.0]))

    def test_constant_path_uses_stable_uniform_progress(self):
        path = scalar_path([2.0, 2.0, 2.0])

        anchors, progress = compress_teacher_path(path, num_student_nodes=5)

        self.assertTrue(torch.isfinite(progress).all())
        torch.testing.assert_close(progress[:, 0], torch.tensor([0.0, 0.5, 1.0]))
        torch.testing.assert_close(anchors, scalar_path([2.0] * 5))

    def test_nonuniform_path_is_interpolated_by_progress(self):
        path = scalar_path([0.0, 1.0, 3.0])

        anchors, _ = compress_teacher_path(path, num_student_nodes=4)

        torch.testing.assert_close(anchors, scalar_path([0.0, 1.0, 2.0, 3.0]))

    def test_single_anchor_uses_final_teacher_state(self):
        path = scalar_path([0.0, 1.0, 3.0])

        anchors, _ = compress_teacher_path(path, num_student_nodes=1)

        torch.testing.assert_close(anchors, scalar_path([3.0]))


class DepthPathLossTest(unittest.TestCase):
    def test_matching_compressed_path_has_zero_losses(self):
        teacher_path = scalar_path([0.0, 1.0, 2.0, 3.0, 4.0])
        student_path = scalar_path([0.0, 2.0, 4.0]).requires_grad_()

        path_loss, macro_loss, anchors = depth_path_losses(student_path, teacher_path)

        torch.testing.assert_close(anchors, student_path.detach())
        torch.testing.assert_close(path_loss, torch.tensor(0.0))
        torch.testing.assert_close(macro_loss, torch.tensor(0.0))

    def test_losses_backpropagate_only_to_student(self):
        teacher_path = scalar_path([0.0, 1.0, 2.0, 4.0]).requires_grad_()
        student_path = scalar_path([0.0, 1.0, 3.0]).requires_grad_()

        path_loss, macro_loss, _ = depth_path_losses(student_path, teacher_path)
        (path_loss + macro_loss).backward()

        self.assertIsNotNone(student_path.grad)
        self.assertIsNone(teacher_path.grad)
        self.assertTrue(torch.isfinite(student_path.grad).all())

    def test_single_student_node_has_zero_macro_loss(self):
        teacher_path = scalar_path([0.0, 1.0, 2.0])
        student_path = scalar_path([0.5])

        _, macro_loss, _ = depth_path_losses(student_path, teacher_path)

        torch.testing.assert_close(macro_loss, torch.tensor(0.0))


class FakeStudent:
    def loss(self, actions, cognition_features):
        return actions.new_tensor(2.0)


class ComputeLossIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.actions = torch.zeros(1, 1, 1)
        self.z = torch.zeros(1, 1, 1)
        self.x0_teacher = torch.zeros_like(self.actions)
        self.x0_student = torch.ones_like(self.actions)
        self.teacher_trajectory = torch.zeros(2, 1, 1, 1)
        self.student_trajectory = torch.ones(2, 1, 1, 1)
        self.teacher_depth_path = scalar_path([0.0, 1.0, 2.0, 4.0])
        self.student_depth_path = scalar_path([0.0, 1.0, 3.0], batch_size=1).requires_grad_()

    def test_path_losses_are_included_in_total(self):
        cfg = DistillationConfig(
            lambda_task=0.0,
            lambda_final=0.0,
            lambda_traj=0.0,
            lambda_path=2.0,
            lambda_macro=3.0,
        )

        losses = compute_loss(
            FakeStudent(),
            self.actions,
            self.z,
            self.x0_teacher,
            self.x0_student,
            self.teacher_trajectory,
            self.student_trajectory,
            cfg,
            teacher_depth_path=self.teacher_depth_path,
            student_depth_path=self.student_depth_path,
        )

        expected = 2.0 * losses["path"] + 3.0 * losses["macro"]
        torch.testing.assert_close(losses["total"], expected)
        losses["total"].backward()
        self.assertIsNotNone(self.student_depth_path.grad)

    def test_enabled_path_loss_requires_both_paths(self):
        cfg = DistillationConfig(lambda_path=1.0)

        with self.assertRaisesRegex(RuntimeError, "depth_path"):
            compute_loss(
                FakeStudent(),
                self.actions,
                self.z,
                self.x0_teacher,
                self.x0_student,
                self.teacher_trajectory,
                self.student_trajectory,
                cfg,
            )

if __name__ == "__main__":
    unittest.main()
