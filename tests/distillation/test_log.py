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
            loss_final=3.0,
            loss_traj=4.0,
            loss_path=5.0,
            loss_macro=6.0,
            grad_norm=7.0,
            lr=1e-4,
        )

        self.assertIn("traj=4.0000", line)
        self.assertIn("path=5.0000", line)
        self.assertIn("macro=6.0000", line)


if __name__ == "__main__":
    unittest.main()
