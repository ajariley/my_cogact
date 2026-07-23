import unittest

import torch

from action_model import create_diffusion
from action_model.action_model import ActionModel
from action_model.models import DiT


class DiTDepthOutputsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = DiT(
            in_channels=3,
            hidden_size=32,
            depth=3,
            num_heads=4,
            token_size=8,
            future_action_window_size=4,
        ).eval()
        self.x = torch.randn(2, 5, 3)
        self.t = torch.tensor([10, 20])
        self.z = torch.randn(2, 1, 8)

    def test_default_forward_is_unchanged(self):
        default_output = self.model(self.x, self.t, self.z)
        final_output, _ = self.model(self.x, self.t, self.z, return_depth_outputs=True)

        self.assertIsInstance(default_output, torch.Tensor)
        torch.testing.assert_close(default_output, final_output)

    def test_depth_output_shape_and_final_node(self):
        final_output, depth_outputs = self.model(
            self.x,
            self.t,
            self.z,
            return_depth_outputs=True,
        )

        self.assertEqual(final_output.shape, (2, 5, 3))
        self.assertEqual(depth_outputs.shape, (3, 2, 5, 3))
        torch.testing.assert_close(depth_outputs[-1], final_output)

    def test_depth_outputs_support_backward(self):
        _, depth_outputs = self.model(
            self.x,
            self.t,
            self.z,
            return_depth_outputs=True,
        )
        depth_outputs.sum().backward()

        self.assertIsNotNone(self.model.blocks[0].attn.qkv.weight.grad)
        self.assertIsNotNone(self.model.final_layer.linear.weight.grad)

    def test_cfg_depth_output_shape(self):
        cfg_x = torch.cat([self.x, self.x], dim=0)
        cfg_t = torch.cat([self.t, self.t], dim=0)
        cfg_z = torch.cat([self.z, torch.zeros_like(self.z)], dim=0)

        final_output, depth_outputs = self.model.forward_with_cfg(
            cfg_x,
            cfg_t,
            cfg_z,
            cfg_scale=1.5,
            return_depth_outputs=True,
        )

        self.assertEqual(final_output.shape, (4, 5, 3))
        self.assertEqual(depth_outputs.shape, (3, 4, 5, 3))
        torch.testing.assert_close(depth_outputs[-1], final_output)


class ActionModelDepthOutputsTest(unittest.TestCase):
    def test_action_model_exposes_depth_outputs(self):
        model = ActionModel(
            token_size=8,
            model_type="DiT-S",
            in_channels=3,
            future_action_window_size=1,
            past_action_window_size=0,
            diffusion_steps=10,
        ).eval()
        x = torch.randn(1, 2, 3)
        timestep = torch.tensor([5])
        z = torch.randn(1, 1, 8)

        final_output, depth_outputs = model(
            x,
            timestep,
            z,
            return_depth_outputs=True,
        )

        self.assertEqual(final_output.shape, (1, 2, 3))
        self.assertEqual(depth_outputs.shape, (6, 1, 2, 3))


class DiffusionDepthPredictionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.diffusion = create_diffusion(
            timestep_respacing="ddim2",
            diffusion_steps=100,
            learn_sigma=False,
        )
        self.model = DiT(
            in_channels=3,
            hidden_size=32,
            depth=3,
            num_heads=4,
            token_size=8,
            future_action_window_size=4,
        ).eval()
        self.noise = torch.randn(2, 5, 3)
        self.z = torch.randn(2, 1, 8)

    def test_depth_eps_to_x0_matches_scalar_formula(self):
        timestep = torch.tensor([0, 1])
        depth_eps = torch.randn(3, *self.noise.shape)

        depth_x0 = self.diffusion._predict_depth_xstart_from_eps(
            self.noise,
            timestep,
            depth_eps,
        )

        for layer in range(depth_eps.shape[0]):
            expected = self.diffusion._predict_xstart_from_eps(
                self.noise,
                timestep,
                depth_eps[layer],
            )
            torch.testing.assert_close(depth_x0[layer], expected)

    def test_ddim_exposes_x0_for_every_depth_node(self):
        outputs = list(
            self.diffusion.ddim_sample_loop_progressive(
                self.model,
                self.noise.shape,
                noise=self.noise,
                clip_denoised=False,
                model_kwargs={"z": self.z, "return_depth_outputs": True},
                device=torch.device("cpu"),
            )
        )

        self.assertEqual(len(outputs), 2)
        for output in outputs:
            self.assertEqual(output["depth_pred_xstart"].shape, (3, 2, 5, 3))
            torch.testing.assert_close(
                output["depth_pred_xstart"][-1],
                output["pred_xstart"],
            )


if __name__ == "__main__":
    unittest.main()
