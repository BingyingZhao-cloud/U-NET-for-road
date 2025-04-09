from torchvision import transforms as T
import numpy as np
import torch
from PIL import Image

# Joint transform 操作，将对 image 和 mask 同步进行
class JointCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask=None):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask

# Resize，同时对 image 和 mask 处理
class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, mask=None):
        image = image.resize(self.size, Image.BILINEAR)
        if mask is not None:
            mask_pil = Image.fromarray(mask)
            mask = mask_pil.resize(self.size, Image.NEAREST)
            mask = np.array(mask)
        return image, mask

# ToTensor，转为 PyTorch tensor
class ToTensor:
    def __call__(self, image, mask=None):
        image = T.ToTensor()(image)
        if mask is not None:
            mask = torch.from_numpy(mask).float().unsqueeze(0)  # shape: [1, H, W]
        return image, mask
