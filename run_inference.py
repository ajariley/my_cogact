"""
CogACT 推理示例脚本
使用方法：
    python3.10 run_inference.py
"""
import os
from PIL import Image
from vla import load_vla
import torch

# 获取 Hugging Face token
hf_token = os.environ.get("HF_TOKEN")

# 加载模型
print("正在加载模型...")
model = load_vla(
    '/data/huangjiaqi/projects/CogACT-Base/checkpoints/CogACT-Base.pt',  # 使用本地下载的模型路径
    hf_token=hf_token,  # 传入 Hugging Face token
    load_for_training=False, 
    action_model_type='DiT-B',              # 选择 ['DiT-S', 'DiT-B', 'DiT-L'] 以匹配模型权重
    future_action_window_size=15,
)

# 可选：使用 bfloat16 以减少内存占用（约30G内存，fp32格式）
# model.vlm = model.vlm.to(torch.bfloat16)

model.to('cuda:0').eval()
print("模型加载完成！")

# 准备输入
image_path = "/data/huangjiaqi/openvla_pic/banana1.jpg"  # 你的图片路径
image = Image.open(image_path)
prompt = "move sponge near apple"  # 输入你的任务描述（根据图片内容修改）

# 预测动作（7-DoF；对于 RT-1 google robot 数据，即 fractal20220817_data，需要反归一化）
print(f"正在预测动作，任务描述: {prompt}")
actions, _ = model.predict_action(
    image,
    prompt,
    unnorm_key='fractal20220817_data',  # 输入你的数据集的 unnorm_key
    cfg_scale=1.5,                       # cfg 从 1.5 到 7 也表现良好
    use_ddim=True,                       # 使用 DDIM 采样
    num_ddim_steps=10,                   # DDIM 采样的步数
)

# 结果：16 步的 7-DoF 动作，形状为 [16, 7]
print(f"预测完成！动作形状: {actions.shape}")
print(f"动作序列:\n{actions}")
