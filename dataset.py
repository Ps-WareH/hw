from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import torch
from skimage import io
class KITTIDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, mask_transform = None, device="cpu"):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.images = os.listdir(image_dir)
        self.device = device  # 设置 GPU 或 CPU

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])
        image = io.imread(img_path)
        mask = io.imread(mask_path)
        
        image = self.transform(image)
        mask = self.mask_transform(mask)
        image = image.to(self.device)
        mask = mask.to(self.device)
        
        return image, mask



class TestDataset(Dataset):
    def __init__(self, image_dir, transform=None,device = None):
        self.device = device
        self.image_dir = image_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = io.imread(img_path)
        if self.transform:
            image = self.transform(image)
        image = image.to(self.device)
        return image, self.images[idx]  # 返回图像及其文件名
