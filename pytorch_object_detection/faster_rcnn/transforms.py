import random
from torchvision.transforms import functional as F
import torch

class Compose(object):
    """组合多个transform函数"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target, mask):
        for t in self.transforms:
            image, target, mask = t(image, target, mask)
        return image, target, mask


class ToTensor(object):
    """将PIL图像转为Tensor"""
    def __call__(self, image, target, mask):
        image = F.to_tensor(image)
        if mask is not None:
            mask = F.to_tensor(mask) #　函数接受PIL Image或numpy.ndarray，将其先由HWC转置为CHW格式，再转为float后每个像素除以255.
        return image, target, mask


class RandomHorizontalFlip(object):
    """随机水平翻转图像以及bboxes"""
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target, mask):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)  # 水平翻转图片
            if mask is not None:
                mask = torch.flip(mask, [2])
            bbox = target["boxes"]
            # bbox: xmin, ymin, xmax, ymax
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]  # 翻转对应bbox坐标信息
            target["boxes"] = bbox
        return image, target, mask
