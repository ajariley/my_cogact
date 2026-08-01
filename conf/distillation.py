

from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class DistillationConfig:
    pretrained_checkpoint: Optional[Path] = None
    teacher_checkpoint: Path = Path("/data/huangjiaqi/projects/CogACT-Base/checkpoints/CogACT-Base.pt")
    data_root_dir: Path = Path("datasets/open-x-embodiment")
    data_mix: str = "bridge"
    output_dir: Path = Path("runs/distillation")
    resume_checkpoint: Optional[Path] = None
    checkpoint_interval: int = 1000
    checkpoint_suffix: str = ""
    metric_header_interval: int = 20
    batch_size: int = 1
    max_batches: Optional[int] = None
    epochs: int = 10
    lr: float = 1e-4  # Deprecated alias; distillation uses base_lr for the optimizer schedule.
    base_lr: float = 1.4e-6
    min_lr: float = 1e-6
    warmup_ratio: float = 0.05
    max_grad_norm: float = 10.0
    num_ddim_steps_teacher: int = 10
    num_ddim_steps_student: int = 4
    action_model_type_teacher: str = "DiT-B"
    action_model_type_student: str = "DiT-B"
    future_action_window_size: int = 15
    past_action_window_size: int = 0
    action_dim: int = 7
    cfg_scale_teacher: float = 1.5  # 仅教师推理时用 CFG，学生不用
    # FSDP 相关参数
    use_fsdp: bool = True
    use_cpu_offload: bool = False  # True 时部分权重留 CPU，易与手写 .to(cuda) 的 batch 设备不一致
    use_bf16: bool = True
    reduce_in_full_precision: bool = True
    use_student_ddp: bool = True
    train_strategy: str = "fsdp"
    # Loss 权重
    lambda_task: float = 0.3
    lambda_final: float = 0.0  # Optional legacy Teacher-final alignment; disabled for strict role separation.
    lambda_final_gt: float = 1.0
    lambda_traj: float = 0.2
    lambda_path: float = 0.0
    lambda_macro: float = 0.0
    exclude_teacher_terminal: bool = True
    refinement_progress_eps: float = 1e-8
    refinement_action_dim_weights: Optional[Tuple[float, ...]] = None
    lambda_neg: float = 0.1
    use_instruction_constraint: bool = True  # 指令敏感：同图+错误指令→z_wrong，L_neg 拉大学生(corr)与(wrong)距离
    log_memory: bool = False
    log_memory_tf: bool = True
    eval_seed: int = 42
    eval_cache_path: Optional[Path] = None
    enable_train_eval: bool = False
    train_eval_interval: int = 1000  #每训练 1000 个 step 才做一次 eval
    train_eval_batches: int = 200
    early_stop_patience: int = 6
    early_stop_min_delta: float = 0
    early_stop_metric: str = "eval/mean_student_teacher_final_mse"

    # Experiment tracking. JSONL is always written locally; add "wandb" or "swanlab" for dashboards.
    trackers: Tuple[str, ...] = ("jsonl",)
    run_id: Optional[str] = None
    wandb_project: str = "cogact-distillation"
    wandb_entity: Optional[str] = None
    swanlab_project: str = "cogact-distillation"
    swanlab_workspace: Optional[str] = None
    swanlab_mode: Optional[str] = None

    # 在 DistillationConfig 中添加（参考 conf/vla.py 的 shuffle_buffer_size）
    shuffle_buffer_size: int = 256000   # 单卡先跑通流程；正式训练可调回 bridge 常用的 256k
    image_aug: bool = False
    load_all_data_for_training: bool = True
