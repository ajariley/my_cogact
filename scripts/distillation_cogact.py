"""
CogACT 扩散头蒸馏脚本

整体流程：
1. 加载教师模型（原版 CogACT，冻结）
2. 加载学生模型（DiT-S 扩散头）
3. 加载数据集 + DataLoader
4. 训练循环：for batch in dataloader
   - 教师 DDIM K 步（带 CFG）+ 记录轨迹、x0
   - 学生 DDIM K 步（无 CFG，单头）（z_corr）+ 可选（z_wrong）
   - Loss = L_task（扩散自身）+ L_distill（轨迹/指令）
   - backward → 只更新学生
5. 保存学生 checkpoint
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, Any

import torch
from torch.utils.data import DataLoader
import draccus

from vla import load_vla, CogACT
from action_model.action_model import ActionModel
from prismatic.vla import get_vla_dataset_and_collator


# -----------------------------------------------------------------------------
# 配置
# -----------------------------------------------------------------------------

@dataclass
class DistillationConfig:
    teacher_checkpoint: Path = Path("/home/huangjiaqi/projects/CogACT-Base/checkpoints/CogACT-Base.pt")
    data_root_dir: Path = Path("datasets/open-x-embodiment")
    data_mix: str = "bridge"
    output_dir: Path = Path("runs/distillation")
    batch_size: int = 4
    max_batches: Optional[int] = None
    epochs: int = 10
    lr: float = 1e-4
    num_ddim_steps_teacher: int = 10
    num_ddim_steps_student: int = 4
    action_model_type_teacher: str = "DiT-B"
    action_model_type_student: str = "DiT-S"
    future_action_window_size: int = 15
    past_action_window_size: int = 0
    action_dim: int = 7
    cfg_scale_teacher: float = 1.5  # 仅教师推理时用 CFG，学生不用
    # Loss 权重
    lambda_task: float = 0.3
    lambda_final: float = 1.0
    lambda_traj: float = 0.5
    lambda_neg: float = 0.1
    use_instruction_constraint: bool = True  # 指令敏感：同图+错误指令→z_wrong，L_neg 拉大学生(corr)与(wrong)距离

    # 在 DistillationConfig 中添加（参考 conf/vla.py 的 shuffle_buffer_size）
    shuffle_buffer_size: int = 256_000   # bridge 用 256k，oxe_magic_soup 用 250k
    image_aug: bool = False
    load_all_data_for_training: bool = True



# -----------------------------------------------------------------------------
# 0. 工具函数
# -----------------------------------------------------------------------------

def _get_cognition_features(
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

# -----------------------------------------------------------------------------
# A. 加载
# -----------------------------------------------------------------------------

def load_teacher(
    checkpoint_path: Path,
    action_model_type: str,
    future_action_window_size: int,
    hf_token: Optional[str] = None,
    **kwargs,
) -> CogACT:
    """
    加载教师 CogACT，eval + 冻结。
    调用：load_vla(...)
    """
    teacher = load_vla(
        str(checkpoint_path),
        hf_token=hf_token,
        load_for_training=False,
        action_model_type=action_model_type,
        future_action_window_size=future_action_window_size,
        **kwargs,
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


def load_student(
    token_size: int,
    action_model_type: str,
    in_channels: int,
    future_action_window_size: int,
    past_action_window_size: int,
    device: torch.device,
) -> ActionModel:
    """
    创建学生 ActionModel（如 DiT-S）。
    调用：ActionModel(token_size=..., model_type=..., ...)
    """
    student = ActionModel(
        token_size=token_size,
        model_type=action_model_type,
        in_channels=in_channels,
        future_action_window_size=future_action_window_size,
        past_action_window_size=past_action_window_size,
    )
    student.create_ddim(ddim_step=4)  # 示例：4 步
    student = student.to(device)
    return student


def load_dataloader(
    teacher,
    data_root_dir: Path,
    data_mix: str,
    batch_size: int,
    **kwargs,
) -> DataLoader:
    """
    加载 dataset + collator，创建 DataLoader。
    调用：get_vla_dataset_and_collator(..., image_transform=teacher.vision_backbone.get_image_transform(), ...)
    """
    vla_dataset, _, collator = get_vla_dataset_and_collator(
        data_root_dir,
        data_mix,
        image_transform=teacher.vision_backbone.get_image_transform(),
        tokenizer=teacher.llm_backbone.get_tokenizer(),
        prompt_builder_fn=teacher.llm_backbone.prompt_builder_fn,
        default_image_resolution=teacher.vision_backbone.default_image_resolution,
        shuffle_buffer_size=kwargs.get("shuffle_buffer_size", 256_000),
        image_aug=kwargs.get("image_aug", False),
        load_all_data_for_training=kwargs.get("load_all_data_for_training", True),
        future_action_window_size=kwargs.get("future_action_window_size", 15),
        past_action_window_size=kwargs.get("past_action_window_size", 0),
        **{k: v for k, v in kwargs.items() if k not in (
            "future_action_window_size", "past_action_window_size",
            "shuffle_buffer_size", "image_aug", "load_all_data_for_training",
        )},
    )
    dataloader = DataLoader(
        vla_dataset,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=0,
    )
    return dataloader


# -----------------------------------------------------------------------------
# B. 教师轨迹生成（带记录）
# 用教师模型当前batch做一次DDIM采样（带CFG），记录轨迹,供loss计算使用
# 返回：{
#        "x0_teacher": ...,
#        "trajectory": [x_T, ..., x_0],
#        "z_corr": ...,
#    }
# -----------------------------------------------------------------------------

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
    z_corr = _get_cognition_features(teacher, batch, device)
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




   


# -----------------------------------------------------------------------------
# D. 损失计算
# -----------------------------------------------------------------------------

def compute_loss(
    teacher_results: Dict[str, Any],
    x0_student_corr: torch.Tensor,
    x0_student_wrong: Optional[torch.Tensor],
    student: ActionModel,
    batch: Dict[str, torch.Tensor],
    cfg: DistillationConfig,
) -> Dict[str, torch.Tensor]:
    """
    总损失 = λ_task * L_task + λ_distill * L_distill
    L_task = action_model.loss(actions, z)  # 扩散自身，真实数据
    L_distill = λ_final*L_final + λ_traj*L_traj + [λ_neg*L_neg]
    L_neg：指令敏感，当 x0_student_wrong 非空时，margin/contrast 使 student(corr) 与 student(wrong) 输出距离足够大
    返回：{"total": ..., "task": ..., "distill": ..., "final": ..., "traj": ..., "neg": ...}
    """
    # TODO: 实现 只做老师和学生的mse
    x0_teacher = teacher_results["x0_teacher"]
    x0_student = x0_student_corr
    loss_mse = torch.nn.functional.mse_loss(x0_teacher, x0_student)
    return {"total": loss_mse, "task": loss_mse, "distill": loss_mse}


    


# -----------------------------------------------------------------------------
# E. 训练循环
# -----------------------------------------------------------------------------

def train_distillation(cfg: DistillationConfig, hf_token: Optional[str] = None) -> None:
    """
    主训练入口。
    流程：load_teacher → load_student → load_dataloader → for batch in dataloader { ... }
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # A. 加载
    teacher = load_teacher(
        cfg.teacher_checkpoint,
        cfg.action_model_type_teacher,
        cfg.future_action_window_size,
        hf_token=hf_token,
    )
    teacher = teacher.to(device)

    token_size = teacher.llm_backbone.llm.lm_head.in_features  # 4096
    student = load_student(
        token_size=token_size,
        action_model_type=cfg.action_model_type_student,
        in_channels=cfg.action_dim,
        future_action_window_size=cfg.future_action_window_size,
        past_action_window_size=cfg.past_action_window_size,
        device=device,
    )

    dataloader = load_dataloader(
        teacher,
        cfg.data_root_dir,
        cfg.data_mix,
        cfg.batch_size,
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
        load_all_data_for_training=cfg.load_all_data_for_training,
        future_action_window_size=cfg.future_action_window_size,
        past_action_window_size=cfg.past_action_window_size,
    )

    optimizer = torch.optim.AdamW(student.parameters(), lr=cfg.lr)

    # B. 训练
    batch_count = 0
    for epoch in range(cfg.epochs):
        for batch in dataloader:
            if cfg.max_batches is not None and batch_count >= cfg.max_batches:
                return#提前停下，以供debug
            batch_count += 1

            # 将 batch 移到 device(Dataloader返回的batch在cpu上，这一步将其放在GPU上)
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else
                {kk: vv.to(device) for kk, vv in v.items()} if isinstance(v, dict) else v
                for k, v in batch.items()
            }

            B = batch["actions"].shape[0]
            T = cfg.future_action_window_size + 1
            C = cfg.action_dim
            noise = torch.randn(B, T, C, device=device)

            # 2. 教师轨迹（带记录）
            with torch.no_grad():
                teacher_results = run_teacher_with_recording(
                    teacher, batch, noise,
                    num_steps=cfg.num_ddim_steps_teacher,
                    cfg_scale=cfg.cfg_scale_teacher,
                    device=device,
                )

            # 3. 学生：用教师输出的 z_corr 做 DDIM 采样
            z_corr = teacher_results["z_corr"]
            x0_student_corr = run_student_ddim(
                student, noise, z_corr, cfg.num_ddim_steps_student, device
            )

            # 4. 学生(z_wrong)，若做指令约束
            x0_student_wrong = None
            if cfg.use_instruction_constraint and B > 1:
                shuffle_idx = torch.randperm(B, device=device)
                batch_wrong = dict(batch)
                batch_wrong["input_ids"] = batch["input_ids"][shuffle_idx]
                batch_wrong["attention_mask"] = batch["attention_mask"][shuffle_idx]
                with torch.no_grad():
                    z_wrong = _get_cognition_features(teacher, batch_wrong, device)
                x0_student_wrong = run_student_ddim(
                    student, noise, z_wrong, cfg.num_ddim_steps_student, device
                )

            # 5. Loss + backward
            loss_dict = compute_loss(
                teacher_results, x0_student_corr, x0_student_wrong, student, batch, cfg
            )
            optimizer.zero_grad()
            loss_dict["total"].backward()
            optimizer.step()

    # 6. 保存学生
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(student.state_dict(), cfg.output_dir / "student_final.pt")


# -----------------------------------------------------------------------------
# 入口
# -----------------------------------------------------------------------------

@draccus.wrap()
def main(cfg: DistillationConfig) -> None:
    hf_token = "hf_xxx" 
    train_distillation(cfg, hf_token=hf_token)


if __name__ == "__main__":
    main()
