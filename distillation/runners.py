
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, Any

import torch
from torch.utils.data import DataLoader
import draccus

from vla import load_vla, CogACT
from action_model.action_model import ActionModel
from prismatic.vla import get_vla_dataset_and_collator
from prismatic.overwatch import initialize_overwatch

from torch.cuda.amp import autocast

overwatch = initialize_overwatch(__name__)

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

    overwatch.info(
        f">>  [distill_debug] teacher.vlm forward "
    )
    with autocast(dtype=torch.bfloat16):
        with torch.no_grad():
            cognition_features = teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_cognition_features=True,
            )
    overwatch.info(
        f">>  [distill_debug] teacher.vlm forward completed"
    )
    if cognition_features.dim() != 3:
        raise RuntimeError(
            f"cognition_features must be 3D [B, 1, D], got shape={tuple(cognition_features.shape)}"
        )
    if cognition_features.size(0) != input_ids.size(0) or cognition_features.size(1) != 1:
        raise RuntimeError(
            "cognition_features shape mismatch: "
            f"features={tuple(cognition_features.shape)}, input_ids={tuple(input_ids.shape)}"
        )
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

    overwatch.info(
        f">> [distill_debug] get_cognition_features batch_shape={tuple(batch['input_ids'].shape)}"
    )
    z_corr = get_cognition_features(teacher, batch, device)

    overwatch.info(
        f">> [distill_debug] zorr_calculating completed z_corr_shape={tuple(z_corr.shape)}"
    )

    # Debug checkpoint: validate z_corr shape early to avoid async CUDA failures later.
    if z_corr.dim() != 3:
        raise RuntimeError(
            f"z_corr must be 3D [B, 1, D], got shape={tuple(z_corr.shape)}"
        )
    B_from_actions = int(batch["actions"].shape[0])
    if z_corr.size(0) != B_from_actions or z_corr.size(1) != 1:
        raise RuntimeError(
            "z_corr shape mismatch: "
            f"z_corr.shape={tuple(z_corr.shape)}, expected [B, 1, D] with B={B_from_actions}"
        )
    print(
        f"[distill_debug] run_teacher_with_recording z_corr_shape={tuple(z_corr.shape)}",
        flush=True,
    )
    B = z_corr.shape[0]                                                 # batch size
    model_dtype = next(teacher.action_model.net.parameters()).dtype     # ActionModel 的 dtype, float32 or float16
    z_corr = z_corr.to(dtype=model_dtype)                               # 将 cognition_features 转换为 ActionModel 的 dtype,避免混合精度导致类型不匹配

    # 2. CFG：双倍 batch，z = [z_corr, uncondition] 
    uncondition = teacher.action_model.net.z_embedder.uncondition # [1, D]

    # 2.1 Debug checkpoint: validate uncondition shape early to avoid async CUDA failures later.
    if uncondition.dim() != 2:
        raise RuntimeError(
            f"uncondition must be 2D [1, D], got shape={tuple(uncondition.shape)}"
        )
    if uncondition.size(0) != 1:
        raise RuntimeError(
            f"uncondition first dim must be 1, got shape={tuple(uncondition.shape)}"
        )
    if uncondition.size(1) != z_corr.size(2):
        raise RuntimeError(
            "uncondition and z_corr hidden dim mismatch: "
            f"uncondition.shape={tuple(uncondition.shape)}, z_corr.shape={tuple(z_corr.shape)}"
        )
    
    # 2
    uncondition = uncondition.unsqueeze(0).expand(B, 1, -1).to(device=device, dtype=model_dtype)
    
    # 2.2 Debug checkpoint: validate uncondition shape early to avoid async CUDA failures later.
    if uncondition.shape != z_corr.shape:
        raise RuntimeError(
            "expanded uncondition shape mismatch with z_corr: "
            f"uncondition.shape={tuple(uncondition.shape)}, z_corr.shape={tuple(z_corr.shape)}"
        )
    overwatch.info(
        f">> [*] [distill_debug] run_teacher_with_recording uncondition_shape={tuple(uncondition.shape)}"
    )
    
    # 2
    z = torch.cat([z_corr, uncondition], dim=0)
    noise_cfg = torch.cat([noise, noise], dim=0).to(device=device, dtype=model_dtype)
    model_kwargs = dict(z=z, cfg_scale=cfg_scale)
    sample_fn = teacher.action_model.net.forward_with_cfg               # 带CFG的forward函数，传入采样循环，用来预测噪声，定义在action_model.py中
    print(
        "[distill_debug] pre_ddim_sampling "
        f"noise_cfg(shape={tuple(noise_cfg.shape)}, dtype={noise_cfg.dtype}, device={noise_cfg.device}) "
        f"z(shape={tuple(z.shape)}, dtype={z.dtype}, device={z.device}) "
        f"cfg_scale={cfg_scale}",
        flush=True,
    )

    _sample_fn_logged = False

    def sample_fn_with_debug(*args, **kwargs):
        nonlocal _sample_fn_logged
        if not _sample_fn_logged:
            x = args[0] if len(args) > 0 and isinstance(args[0], torch.Tensor) else None
            t = args[1] if len(args) > 1 and isinstance(args[1], torch.Tensor) else None
            z_in = kwargs.get("z", None)
            z_msg = (
                f"shape={tuple(z_in.shape)} dtype={z_in.dtype} device={z_in.device}"
                if isinstance(z_in, torch.Tensor)
                else str(type(z_in))
            )
            x_msg = (
                f"shape={tuple(x.shape)} dtype={x.dtype} device={x.device}"
                if isinstance(x, torch.Tensor)
                else "N/A"
            )
            t_msg = (
                f"shape={tuple(t.shape)} dtype={t.dtype} device={t.device}"
                if isinstance(t, torch.Tensor)
                else "N/A"
            )
            print(
                "[distill_debug] enter_forward_with_cfg "
                f"x({x_msg}) t({t_msg}) z({z_msg})",
                flush=True,
            )
            _sample_fn_logged = True
        return sample_fn(*args, **kwargs)
    # debug结束

    # 3. DDIM 采样
    teacher.action_model.create_ddim(ddim_step=num_steps)
    trajectory = []
    for out in teacher.action_model.ddim_diffusion.ddim_sample_loop_progressive(# 采样循环，定义在ddim_diffusion.py中
        sample_fn_with_debug,
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
        grad_enabled=True,
    )
    return x0_student
