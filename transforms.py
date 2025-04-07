import numpy as np
from PIL import Image
import torch
from torchvision import transforms as T

class JointCompose:
    """
    同时对 image 和 mask 应用一系列变换
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask

class Resize:
    """
    将 PIL Image 和 numpy mask 一起 resize
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, image, mask):
        # image: PIL.Image, mask: numpy.ndarray
        image = image.resize(self.size, resample=Image.BILINEAR)
        mask_pil = Image.fromarray(mask)
        mask_pil = mask_pil.resize(self.size, resample=Image.NEAREST)
        mask = np.array(mask_pil)
        return image, mask

class ToTensor:
    """
    将 PIL Image 转为 Tensor,将 numpy mask 转为 Tensor
    """
    def __call__(self, image, mask):
        image = T.ToTensor()(image)                # [C, H, W], float [0,1]
        mask  = torch.from_numpy(mask).unsqueeze(0).float()  # [1, H, W]
        return image, mask
