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
import importlib.util
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")  # 保持已有设置
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"   # TF 按需申请显存，而不是一次性占满
# 或者更彻底：os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Baseline snapshot before torch / tensorflow / prismatic / distillation __init__ (no PyTorch import).
# Disable with: COGACT_LOG_MEMORY_EARLY=0
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if os.environ.get("COGACT_LOG_MEMORY_EARLY", "1") != "0":
    _mem_path = _PROJECT_ROOT / "distillation" / "memory_log.py"
    _spec = importlib.util.spec_from_file_location("_cogact_memory_log", _mem_path)
    if _spec and _spec.loader:
        _mem_mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mem_mod)
        _mem_mod.log_memory_process_start("process_start_before_heavy_imports")

from dataclasses import dataclass
from typing import Dict, Optional, Any

import torch
from torch.utils.data import DataLoader
import draccus

from vla import load_vla, CogACT
from action_model.action_model import ActionModel
from prismatic.vla import get_vla_dataset_and_collator

sys.path.insert(0, str(_PROJECT_ROOT))
# 配置，工具函数导入
from distillation import train_distillation
from distillation.memory_log import log_memory
from conf import DistillationConfig


# -----------------------------------------------------------------------------
# 入口
# -----------------------------------------------------------------------------

@draccus.wrap()
def main(cfg: DistillationConfig) -> None:
    if cfg.log_memory:
        log_memory("after_imports", log_tf=cfg.log_memory_tf)

    hf_token = "hf_xxx"
    train_distillation(cfg, hf_token=hf_token)


if __name__ == "__main__":
    try:
        print("[distillation_cogact] interpreter started", flush=True)
        main()
    except BaseException:
        import traceback

        traceback.print_exc()
        raise
