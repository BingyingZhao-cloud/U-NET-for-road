import os
import torch
import numpy as np
import random
from torchvision.utils import save_image

def save_checkpoint(model, path):
    """保存模型参数"""
    torch.save(model.state_dict(), path)
    print(f"[INFO] 模型已保存到: {path}")

def load_checkpoint(model, path, device):
    """加载模型参数"""
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        print(f"[INFO] 模型已从 {path} 成功加载")
    else:
        print(f"[WARNING] 模型路径不存在: {path}")

def set_seed(seed=42):
    """设置所有随机种子，确保实验可复现"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def ensure_dir(path):
    """确保路径存在"""
    if not os.path.exists(path):
        os.makedirs(path)

def save_prediction_image(pred_tensor, save_path):
    """
    将模型输出的预测图像保存为 PNG 图（默认假设为 [1, H, W] 或 [B, 1, H, W]）
    """
    if pred_tensor.dim() == 4:
        for i, pred in enumerate(pred_tensor):
            save_image(pred, os.path.join(save_path, f"pred_{i}.png"))
    else:
        save_image(pred_tensor, save_path)
