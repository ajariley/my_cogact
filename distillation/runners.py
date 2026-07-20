
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

  
    with autocast(dtype=torch.bfloat16):
        with torch.no_grad():
            cognition_features = teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_cognition_features=True,
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
    num_steps: int,  #实际上要跑的步数
    record_timesteps: list[int],  #最终记录的内容
    cfg_scale: float,
    device: torch.device,
) -> Dict[str, Any]:
    """
    教师 DDIM K 步（带 CFG），记录轨迹。
    思路：forward_with_cfg(eps_cond, eps_uncond, scale) + ddim_sample_loop_progressive。
    返回：{
        "x0_teacher": ...,
        "trajectory": [K, B, T, C],
        "depth_x0_trajectory": [K_full, L, B, T, C],
        "depth_x0_path": [K_full * L, B, T, C],
        "timesteps": [original diffusion timestep, ...],
        "z_corr": ...,
    }
    """
    # 1. VLM 前向得到 cognition_features
 
    z_corr = get_cognition_features(teacher, batch, device)


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
    
    B = z_corr.shape[0]                                                 # batch size
    model_dtype = next(teacher.action_model.net.parameters()).dtype     # ActionModel 的 dtype, float32 or float16
    z_corr = z_corr.to(dtype=model_dtype)                               # 将 cognition_features 转换为 ActionModel 的 dtype,避免混合精度导致类型不匹配

    # 2. CFG：双倍 batch，z = [z_corr, uncondition] 
    uncondition = teacher.action_model.net.z_embedder.uncondition # [1, D]

    
    # 2
    uncondition = uncondition.unsqueeze(0).expand(B, 1, -1).to(device=device, dtype=model_dtype)
    
    
    # 2
    z = torch.cat([z_corr, uncondition], dim=0)
    noise_cfg = torch.cat([noise, noise], dim=0).to(device=device, dtype=model_dtype)
    model_kwargs = dict(z=z, cfg_scale=cfg_scale, return_depth_outputs=True)
    sample_fn = teacher.action_model.net.forward_with_cfg               # 带CFG的forward函数，传入采样循环，用来预测噪声，定义在action_model.py中

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
            
            _sample_fn_logged = True
        return sample_fn(*args, **kwargs)
    # debug结束

    # 3. DDIM 采样
    teacher.action_model.create_ddim(ddim_step=num_steps)
    teacher_timesteps = _ddim_original_timesteps(teacher.action_model.ddim_diffusion)
    missing_timesteps = [t for t in record_timesteps if t not in teacher_timesteps]
    if missing_timesteps:
        raise RuntimeError(
            "teacher DDIM timeline does not cover requested student timesteps: "
            f"missing={missing_timesteps}, teacher={teacher_timesteps}, requested={record_timesteps}"
        )
    record_timestep_set = set(record_timesteps)

    trajectory = []
    depth_x0_trajectory = []
    for timestep, out in zip(
        teacher_timesteps,
        teacher.action_model.ddim_diffusion.ddim_sample_loop_progressive(# 采样循环，定义在ddim_diffusion.py中
        sample_fn_with_debug,
        noise_cfg.shape,
        noise=noise_cfg,
        clip_denoised=False,
        model_kwargs=model_kwargs,
        device=device,
        progress=False,
        eta=0.0,
        ),
    ):
        depth_pred_xstart = out["depth_pred_xstart"]
        if depth_pred_xstart is None:
            raise RuntimeError("teacher DDIM step did not return per-block x0 predictions")
        depth_pred_xstart, _ = depth_pred_xstart.chunk(2, dim=1)
        depth_x0_trajectory.append(depth_pred_xstart)
        if timestep in record_timestep_set:
            sample, _ = out["sample"].chunk(2, dim=0)  # 去掉 CFG 的 null 部分
            trajectory.append(sample)

    if len(trajectory) != len(record_timesteps):
        raise RuntimeError(
            "teacher recorded trajectory length mismatch: "
            f"got={len(trajectory)}, expected={len(record_timesteps)}, requested={record_timesteps}"
        )

    teacher_trajectory = torch.stack(trajectory, dim=0)
    teacher_depth_x0_trajectory = torch.stack(depth_x0_trajectory, dim=0)
    teacher_depth_x0_path = teacher_depth_x0_trajectory.flatten(0, 1)
    return {
        "x0_teacher": teacher_trajectory[-1],
        "trajectory": teacher_trajectory,
        "depth_x0_trajectory": teacher_depth_x0_trajectory,
        "depth_x0_path": teacher_depth_x0_path,
        "depth_timesteps": teacher_timesteps,
        "timesteps": record_timesteps,
        "teacher_full_timesteps": teacher_timesteps,
        "z_corr": z_corr,
    }


# -----------------------------------------------------------------------------
# C. 学生 DDIM 少步
# -----------------------------------------------------------------------------

def _ddim_original_timesteps(ddim_diffusion) -> list[int]:
    timestep_map = getattr(ddim_diffusion, "timestep_map", None)
    if timestep_map is None:
        timestep_map = list(range(ddim_diffusion.num_timesteps))
    return [int(timestep_map[i]) for i in range(ddim_diffusion.num_timesteps - 1, -1, -1)]


def get_student_timesteps(student: ActionModel, num_steps: int) -> list[int]:
    if student.ddim_diffusion is None or student.ddim_diffusion.num_timesteps != num_steps:
        student.create_ddim(ddim_step=num_steps)
    return _ddim_original_timesteps(student.ddim_diffusion)


def run_student_ddim_with_recording(
    student: ActionModel,
    noise: torch.Tensor,
    z: torch.Tensor,
    num_steps: int,
    device: torch.device,
) -> Dict[str, Any]:
    """
    学生 DDIM K 步，无 CFG（单头），记录学生实际经过的 sample trajectory。
    z 即教师 VLM 输出的 z_corr，作为条件传入。
    返回：{
        "x0_student": [B, T, C],
        "trajectory": [K, B, T, C],
        "depth_x0_trajectory": [K, L, B, T, C],
        "depth_x0_path": [K * L, B, T, C],
        "timesteps": [original diffusion timestep, ...],
    }
    """
    if student.ddim_diffusion is None or student.ddim_diffusion.num_timesteps != num_steps:
        student.create_ddim(ddim_step=num_steps)
    model_kwargs = dict(z=z, return_depth_outputs=True)
    sample_fn = student.net

    trajectory = []
    depth_x0_trajectory = []
    for out in student.ddim_diffusion.ddim_sample_loop_progressive(
        sample_fn,
        noise.shape,
        noise=noise,
        clip_denoised=False,
        model_kwargs=model_kwargs,
        device=device,
        progress=False,
        eta=0.0,
        grad_enabled=True,
    ):
        trajectory.append(out["sample"])
        depth_pred_xstart = out["depth_pred_xstart"]
        if depth_pred_xstart is None:
            raise RuntimeError("student DDIM step did not return per-block x0 predictions")
        depth_x0_trajectory.append(depth_pred_xstart)

    if not trajectory:
        raise RuntimeError("student DDIM produced an empty trajectory")

    student_depth_x0_trajectory = torch.stack(depth_x0_trajectory, dim=0)
    return {
        "x0_student": trajectory[-1],
        "trajectory": torch.stack(trajectory, dim=0),
        "depth_x0_trajectory": student_depth_x0_trajectory,
        "depth_x0_path": student_depth_x0_trajectory.flatten(0, 1),
        "depth_timesteps": _ddim_original_timesteps(student.ddim_diffusion),
        "timesteps": _ddim_original_timesteps(student.ddim_diffusion),
    }
