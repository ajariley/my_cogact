#!/bin/bash
# 单卡训练脚本（RTX 5090, 32GB）
# 使用方法: bash train_single_gpu.sh

# 设置 Hugging Face token（如果还没设置）
export HF_TOKEN=${HF_TOKEN:-"hf_你的token"}

# 设置代理（如果需要）
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890

# 训练参数配置
PRETRAINED_CHECKPOINT="/home/huangjiaqi/projects/CogACT-Base/checkpoints/CogACT-Base.pt"  # 或使用 "CogACT/CogACT-Base"
DATA_ROOT_DIR="/path/to/your/dataset"  # 替换为你的数据集路径
RUN_ROOT_DIR="/home/huangjiaqi/projects/CogACT/runs"  # 训练日志和checkpoint保存路径
RUN_ID="cogact_single_gpu_$(date +%Y%m%d_%H%M%S)"  # 运行ID

# 单卡训练命令
torchrun --standalone --nnodes 1 --nproc-per-node 1 scripts/train.py \
  --pretrained_checkpoint "${PRETRAINED_CHECKPOINT}" \
  --vla.type prism-dinosiglip-224px+oxe+diffusion \
  --vla.data_mix bridge \
  --vla.expected_world_size 1 \
  --vla.global_batch_size 8 \
  --vla.per_device_batch_size 8 \
  --vla.learning_rate 2e-5 \
  --data_root_dir "${DATA_ROOT_DIR}" \
  --run_root_dir "${RUN_ROOT_DIR}" \
  --run_id "${RUN_ID}" \
  --image_aug True \
  --wandb_project cogact_training \
  --wandb_entity your_wandb_entity \
  --save_interval 1000 \
  --repeated_diffusion_steps 8 \
  --future_action_window_size 15 \
  --action_model_type DiT-B \
  --is_resume False

