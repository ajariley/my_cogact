import unittest

from distillation.log import format_metric_line


class MetricLineTest(unittest.TestCase):
    def test_depth_losses_are_accepted_and_rendered(self):
        line = format_metric_line(
            step=1,
            epoch=1,
            epochs=10,
            loss_total=1.0,
            loss_task=2.0,
            loss_final_gt=3.0,
            loss_final_teacher=3.5,
            loss_traj_teacher=4.0,
            loss_path_teacher=5.0,
            loss_macro_teacher=6.0,
            grad_norm=7.0,
            lr=1e-4,
        )

        self.assertIn("final_gt=3.0000", line)
        self.assertIn("traj_teacher=4.0000", line)
        self.assertIn("path_teacher=5.0000", line)
        self.assertIn("macro_teacher=6.0000", line)


if __name__ == "__main__":
    unittest.main()
