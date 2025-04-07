import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from config import TRAIN_IMAGE_DIR, TRAIN_MASK_DIR, BINARIZATION_THRESHOLD, IMAGE_SIZE
from transforms import JointCompose, Resize, ToTensor

# 定义 joint_transform
joint_transform = JointCompose([
    Resize(IMAGE_SIZE),
    ToTensor(),
])

class RoadSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=joint_transform, threshold=BINARIZATION_THRESHOLD):
        """
        Args:
            image_dir (str): 卫星图像目录
            mask_dir  (str): 标签目录
            transform (callable): 同时处理 image 和 mask 的变换
            threshold (int): 二值化阈值
        """
        self.image_dir = image_dir
        self.mask_dir  = mask_dir
        self.image_list = sorted([f for f in os.listdir(image_dir) if f.endswith('_sat.jpg')])
        self.transform = transform
        self.threshold = threshold

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        mask_name  = image_name.replace('_sat.jpg', '_mask.png')

        image_path = os.path.join(self.image_dir, image_name)
        mask_path  = os.path.join(self.mask_dir,  mask_name)

        # 读取图像与标签
        image = Image.open(image_path).convert("RGB")
        mask  = Image.open(mask_path).convert("L")  # 灰度

        # 二值化处理
        mask_np = np.array(mask)
        mask_bin = (mask_np >= self.threshold).astype(np.uint8)

        # 应用 joint_transform
        image, mask_bin = self.transform(image, mask_bin)

        return image, mask_bin

# 测试数据集加载
if __name__ == '__main__':
    from torch.utils.data import DataLoader
    dataset = RoadSegmentationDataset(
        image_dir=TRAIN_IMAGE_DIR,
        mask_dir=TRAIN_MASK_DIR
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2)
    imgs, masks = next(iter(loader))
    print('images:', imgs.shape)  
    print('masks :', masks.shape) 
