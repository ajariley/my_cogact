
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, Any

import torch
from torch.utils.data import DataLoader
import draccus

from vla import load_vla, CogACT
from action_model.action_model import ActionModel
from prismatic.vla import get_vla_dataset_and_collator

def get_cognition_features(
    teacher: CogACT,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """
    从 batch 经 VLM 得 cognition_features z [B, 1, D]。
    参考 cogactvla.py forward 第 114-145 行。
    """
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    pixel_values = batch["pixel_values"]
    if isinstance(pixel_values, dict):
        pixel_values = {k: v.to(device) for k, v in pixel_values.items()}
    else:
        pixel_values = pixel_values.to(device)

    with torch.no_grad():
        output = teacher.vlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            output_hidden_states=True,
        )

    last_hidden = output.hidden_states[-1]
    if teacher.vlm.vision_backbone.featurizer is not None:
        num_patch = teacher.vlm.vision_backbone.featurizer.patch_embed.num_patches
    elif hasattr(teacher.vlm.vision_backbone, "siglip_featurizer") and teacher.vlm.vision_backbone.siglip_featurizer is not None:
        num_patch = teacher.vlm.vision_backbone.siglip_featurizer.patch_embed.num_patches
    else:
        raise ValueError("No vision backbone found")

    last_hidden = last_hidden[:, num_patch:]
    cumulative_sum = attention_mask.cumsum(dim=1)
    last_true_indices = (cumulative_sum == cumulative_sum.max(dim=1, keepdim=True)[0]).float().argmax(dim=1)
    expanded_indices = last_true_indices.unsqueeze(-1).expand(-1, last_hidden.size(-1))
    cognition_features = last_hidden.gather(1, expanded_indices.unsqueeze(1))  # [B, 1, D]
    return cognition_features



    
def run_teacher_with_recording(
    teacher,
    batch: Dict[str, torch.Tensor],
    noise: torch.Tensor,
    num_steps: int,
    cfg_scale: float,
    device: torch.device,
) -> Dict[str, Any]:
    """
    教师 DDIM K 步（带 CFG），记录轨迹。
    思路：forward_with_cfg(eps_cond, eps_uncond, scale) + ddim_sample_loop_progressive。
    返回：{
        "x0_teacher": ...,
        "trajectory": [x_T, ..., x_0],
        "z_corr": ...,
    }
    """
    # 1. VLM 前向得到 cognition_features
    z_corr = get_cognition_features(teacher, batch, device)
    B = z_corr.shape[0]                                                 # batch size
    model_dtype = next(teacher.action_model.net.parameters()).dtype     # ActionModel 的 dtype, float32 or float16
    z_corr = z_corr.to(model_dtype)                                     # 将 cognition_features 转换为 ActionModel 的 dtype,避免混合精度导致类型不匹配

    # 2. CFG：双倍 batch，z = [z_corr, uncondition] 
    uncondition = teacher.action_model.net.z_embedder.uncondition # [1, D]
    uncondition = uncondition.unsqueeze(0).expand(B, 1, -1).to(device)
    z = torch.cat([z_corr, uncondition], dim=0)
    noise_cfg = torch.cat([noise, noise], dim=0)
    model_kwargs = dict(z=z, cfg_scale=cfg_scale)
    sample_fn = teacher.action_model.net.forward_with_cfg               # 带CFG的forward函数，传入采样循环，用来预测噪声，定义在action_model.py中

    # 3. DDIM 采样
    teacher.action_model.create_ddim(ddim_step=num_steps)
    trajectory = []
    for out in teacher.action_model.ddim_diffusion.ddim_sample_loop_progressive(# 采样循环，定义在ddim_diffusion.py中
        sample_fn,
        noise_cfg.shape,
        noise=noise_cfg,
        clip_denoised=False,
        model_kwargs=model_kwargs,
        device=device,
        progress=False,
        eta=0.0,
    ):
        trajectory.append(out)

    x0_teacher = trajectory[-1]["sample"]
    x0_teacher, _ = x0_teacher.chunk(2, dim=0)  # 去掉 CFG 的 null 部分

    return {"x0_teacher": x0_teacher, "trajectory": trajectory, "z_corr": z_corr}


# -----------------------------------------------------------------------------
# C. 学生 DDIM 少步
# -----------------------------------------------------------------------------

def run_student_ddim(
    student: ActionModel,
    noise: torch.Tensor,
    z: torch.Tensor,
    num_steps: int,
    device: torch.device,
) -> torch.Tensor:
    """
    学生 DDIM K 步，无 CFG（单头）。
    z 即教师 VLM 输出的 z_corr，作为条件传入。
    返回：x0_student [B, T, C]
    """
    if student.ddim_diffusion is None or student.ddim_diffusion.num_timesteps != num_steps:
        student.create_ddim(ddim_step=num_steps)
    model_kwargs = dict(z=z)
    sample_fn = student.net.forward
    x0_student = student.ddim_diffusion.ddim_sample_loop(
        sample_fn,
        noise.shape,
        noise=noise,
        clip_denoised=False,
        model_kwargs=model_kwargs,
        device=device,
        progress=False,
        eta=0.0,
    )
    return x0_student

