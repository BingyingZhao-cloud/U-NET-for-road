import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from config import TRAIN_IMAGE_DIR, TRAIN_MASK_DIR, BINARIZATION_THRESHOLD, IMAGE_SIZE


class RoadSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform = None,threshold = BINARIZATION_THRESHOLD):
        """
        image_dir (str):图像存放目录, eg: TRAIN_IMAGE_DIR
        label_dir (str):标签存放目录路径 eg:TRAIN_MASK_DIR
        transform (callable,optional):数据增强或与处理函数. 输入为PIL.Image和numpy数组,返回处理后的图像和标签。
        threshold (int):二值化阙值,默认使用config中的BINARIZATION_THRESHOLD(128)
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir 

        #筛选出所有以_sat.jpg结尾的图像文件，确保图像和mask能对应上
        self.image_list = sorted([f for f in os.listdir(image_dir) if f.endswith('_sat.jpg')])
        self.transform = transform
        self.threshold = threshold

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        #获取图像文件名和对应的mask文件名
        image_name = self.image_list[idx]
        mask_name = image_name.replace('_sat.jpg','_mask.png')

        #构建图像和mask的完整路径
        image_path = os.path.join(self.image_dir,image_name)
        mask_path = os.path.join(self.mask_dir,mask_name)

        #读取图像(转为RPG)
        image = Image.open(image_path).convert("RGB")
        image = image.resize(IMAGE_SIZE)

        #读取mask(灰度图)
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize(IMAGE_SIZE)

        #将mask转换为numpy数组，并进行二值化处理：
        #大于等于阙值（128）的像素设为1（道路），小于阙值设为0（背景）
        mask_np = np.array(mask)
        mask_bin = (mask_np >= self.threshold).astype(np.uint8)

        #如果定义了transform函数，对图像进行数据增强或者预处理
        if self.transform:
            image, mask_bin = self.transform(image,mask_bin)
        else:
            #默认转换为tensor
            image = transforms.ToTensor()(image)
            #为mask添加通道维度，并转换为tensor
            mask_bin = torch.form_numpy(mask_bin).unsqueeze(0).float()

        return image,mask_bin
    
def simple_transform(image,mask):
        image = transforms.ToTensor()(image)
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        return image,mask
    
if __name__ == '__main__':
    dataset = RoadSegmentationDataset(
          image_dir=TRAIN_IMAGE_DIR,
          mask_dir=TRAIN_MASK_DIR,
          transform=simple_transform
    )


    #打印第一个样本的图像和mask形状
    img,msk = dataset[0]
    print('图像形状: ',img.shape)
    print('mask形状:',msk.shape)