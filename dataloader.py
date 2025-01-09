import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

# 数据增强函数
def rotate(img, msk, degrees=(-15, 15), p=0.5):
    if torch.rand(1).item() < p:
        degree = np.random.uniform(*degrees)
        img = img.rotate(degree, Image.NEAREST)
        msk = msk.rotate(degree, Image.NEAREST)
    return img, msk

def horizontal_flip(img, msk, p=0.5):
    if torch.rand(1).item() < p:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        msk = msk.transpose(Image.FLIP_LEFT_RIGHT)
    return img, msk

def vertical_flip(img, msk, p=0.5):
    if torch.rand(1).item() < p:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        msk = msk.transpose(Image.FLIP_TOP_BOTTOM)
    return img, msk

def augment(img, msk):
    img, msk = horizontal_flip(img, msk)
    img, msk = vertical_flip(img, msk)
    img, msk = rotate(img, msk)
    return img, msk

# 自定义数据集类
class MyDataset(Dataset):
    """
    自定义数据集类，从图像文件加载数据（灰度图）并统一调整大小为 256x256。

    参数：
        images_dir (str): 图像文件夹路径。
        masks_dir (str): 掩码文件夹路径。
        transform (bool): 是否应用数据增强。
        transform_fn (callable, optional): 其他转换函数（如归一化等）。
    """
    def __init__(self, images_dir, masks_dir, transform=False, transform_fn=None):
        super().__init__()
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.transform_fn = transform_fn

        # 获取所有图像和掩码文件名，并排序确保对应关系
        self.image_files = sorted([f for f in os.listdir(images_dir) if self._is_image_file(f)])
        self.mask_files = sorted([f for f in os.listdir(masks_dir) if self._is_image_file(f)])

        assert len(self.image_files) == len(self.mask_files), "图像和掩码的数量不匹配"

        # 定义统一的调整大小转换
        self.resize_img = transforms.Resize((224, 224), interpolation=Image.BILINEAR)
        self.resize_msk = transforms.Resize((224, 224), interpolation=Image.NEAREST)

    def _is_image_file(self, filename):
        """检查文件是否是图像格式"""
        IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        return filename.lower().endswith(IMG_EXTENSIONS)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 获取图像和掩码的完整路径
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        mask_path = os.path.join(self.masks_dir, self.mask_files[idx])

        # 打开图像和掩码（均为灰度图）
        img = Image.open(img_path).convert('L')  # 灰度图
        msk = Image.open(mask_path).convert('L')  # 灰度图

        # 统一调整大小为 256x256
        img = self.resize_img(img)
        msk = self.resize_msk(msk)

        # 应用数据增强
        if self.transform:
            img, msk = augment(img, msk)

        # 转换为张量
        img = transforms.ToTensor()(img)  # 形状为 [1, 256, 256]
        msk = transforms.ToTensor()(msk)  # 形状为 [1, 256, 256]

        # 可选的其他转换
        if self.transform_fn:
            img = self.transform_fn(img)
            msk = self.transform_fn(msk)

        return img, msk