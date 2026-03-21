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

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# 配置，工具函数导入
from distillation import train_distillation
from conf import DistillationConfig


# -----------------------------------------------------------------------------
# 入口
# -----------------------------------------------------------------------------

@draccus.wrap()
def main(cfg: DistillationConfig) -> None:
    hf_token = "hf_xxx" 
    train_distillation(cfg, hf_token=hf_token)


if __name__ == "__main__":
    main()
