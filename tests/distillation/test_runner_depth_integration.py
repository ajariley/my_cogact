import unittest

import torch

from action_model.action_model import ActionModel
from distillation.runners import run_student_ddim_with_recording


class StudentDepthRunnerIntegrationTest(unittest.TestCase):
    def test_two_step_six_layer_path_supports_backward(self):
        torch.manual_seed(0)
        student = ActionModel(
            token_size=8,
            model_type="DiT-S",
            in_channels=3,
            future_action_window_size=1,
            past_action_window_size=0,
        ).eval()
        noise = torch.randn(1, 2, 3)
        cognition_features = torch.randn(1, 1, 8)

        output = run_student_ddim_with_recording(
            student,
            noise,
            cognition_features,
            num_steps=2,
            device=torch.device("cpu"),
        )

        self.assertEqual(output["trajectory"].shape, (2, 1, 2, 3))
        self.assertEqual(output["depth_x0_trajectory"].shape, (2, 6, 1, 2, 3))
        self.assertEqual(output["depth_x0_path"].shape, (12, 1, 2, 3))
        self.assertTrue(output["depth_x0_path"].requires_grad)
        for timestep_index in range(2):
            torch.testing.assert_close(
                output["depth_x0_path"][timestep_index * 6 : (timestep_index + 1) * 6],
                output["depth_x0_trajectory"][timestep_index],
            )

        output["depth_x0_path"].square().mean().backward()
        self.assertIsNotNone(student.net.final_layer.linear.weight.grad)


if __name__ == "__main__":
    unittest.main()
