import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from config import TRAIN_IMAGE_DIR, TRAIN_MASK_DIR, BINARIZATION_THRESHOLD, IMAGE_SIZE
from transforms import JointCompose, Resize, ToTensor

# 定义 joint_transform：同时对 image 和 mask 进行 Resize 和 ToTensor 转换
joint_transform = JointCompose([
    Resize(IMAGE_SIZE),
    ToTensor(),
])

class RoadSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=joint_transform, threshold=BINARIZATION_THRESHOLD):
        """
        Args:
            image_dir (str): 卫星图像目录，文件后缀应为 '_sat.jpg'
            mask_dir  (str): 标签目录，标签文件后缀应为 '_mask.png'
            transform (callable): 同时处理 image 和 mask 的变换
            threshold (int): 二值化阈值
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        # 筛选出以 '_sat.jpg' 结尾的文件列表
        self.image_list = sorted([f for f in os.listdir(image_dir) if f.endswith('_sat.jpg')])
        self.transform = transform
        self.threshold = threshold
        self.has_mask = mask_dir is not None

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        image_path = os.path.join(self.image_dir, image_name)

        image = Image.open(image_path).convert("RGB")


        if self.has_mask:
            mask_name = image_name.replace('_sat.jpg', '_mask.png')
            mask_path = os.path.join(self.mask_dir, mask_name)

            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask file not found: {mask_path}")
            
            mask = Image.open(mask_path).convert("L")
            mask_np = np.array(mask)
            mask_bin = (mask_np >= self.threshold).astype(np.uint8)

            image, mask_bin = self.transform(image, mask_bin)
            return image, mask_bin
        else:
            # 无标签时只返回图像
            image, _ = self.transform(image, None)
            return image


# 测试数据集加载
if __name__ == '__main__':
    from torch.utils.data import DataLoader
    # 使用 config 中的路径加载数据
    dataset = RoadSegmentationDataset(
        image_dir=TRAIN_IMAGE_DIR,
        mask_dir=TRAIN_MASK_DIR
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2)
    imgs, masks = next(iter(loader))
    print('images:', imgs.shape)  # 例如：[2, 3, H, W]
    print('masks :', masks.shape)  # 例如：[2, 1, H, W]
