"""
蒸馏专用 FSDP 包装模块。

仅负责用 FSDP 包装教师 CogACT，降低显存（BF16 + CPU Offload）。
不实现 optimizer、checkpoint、gradient checkpointing。
"""

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    CPUOffload,
)

from prismatic.overwatch import initialize_overwatch
from vla import CogACT

overwatch = initialize_overwatch(__name__)


def wrap_teacher_with_fsdp(
    teacher: CogACT,
    use_cpu_offload: bool = False,
    use_bf16: bool = True,
    reduce_in_full_precision: bool = True,
) -> CogACT:
    """
    用 FSDP 包装教师，返回包装后的 teacher。
    需在 torchrun 下调用。
    """
    # Step 1: MixedPrecision
    if use_bf16:
        reduce_dtype = torch.float32 if reduce_in_full_precision else torch.bfloat16
        fsdp_precision_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=reduce_dtype,
            buffer_dtype=torch.bfloat16,
        )
        overwatch.info(
            f"[Mixed Precision Enabled] Parameter dtype: {fsdp_precision_policy.param_dtype}, "
            f"Reduce dtype: {fsdp_precision_policy.reduce_dtype}, "
            f"Buffer dtype: {fsdp_precision_policy.buffer_dtype}"
        )
    else:
        fsdp_precision_policy = MixedPrecision(
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        )
        overwatch.info(
            f"[Mixed Precision Disabled] All dtypes set to FP32: "
            f"Parameter={fsdp_precision_policy.param_dtype}, "
            f"Reduce={fsdp_precision_policy.reduce_dtype}, "
            f"Buffer={fsdp_precision_policy.buffer_dtype}"
        )

    # Step 2: Vision backbone 转 half（在 FSDP wrap 前）
    if use_bf16:
        overwatch.info("Casting Vision Backbone to *Half Precision* via `.to(dtype=...)`")
        teacher.vision_backbone.to(dtype=teacher.vision_backbone.half_precision_dtype)

    # Step 3: CPUOffload
    cpu_offload = CPUOffload(offload_params=use_cpu_offload)
    overwatch.info(f"CPU Offload: offload_params={use_cpu_offload}")

    # 单卡时 FSDP 退化为 NO_SHARD，无跨卡分片
    if dist.is_initialized() and dist.get_world_size() == 1:
        overwatch.info(
            "World size=1: FSDP 将退化为 NO_SHARD，无跨卡分片。若 OOM，可尝试 use_fsdp=False + use_bf16=True。"
        )

    # Step 4: FSDP wrap
    teacher = FSDP(
        teacher,
        auto_wrap_policy=teacher.get_fsdp_wrapping_policy(),
        mixed_precision=fsdp_precision_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        use_orig_params=True,
        cpu_offload=cpu_offload,
    )

    return teacher
