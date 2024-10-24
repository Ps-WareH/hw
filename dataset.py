from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
from skimage import io
class KITTIDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, mask_transform = None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])
        image = io.imread(img_path)
        mask = io.imread(mask_path)
    
        image = self.transform(image)
        mask = self.mask_transform(mask)
        return image, mask



class TestDataset(Dataset):
    def __init__(self, image_dir, transform=None):
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
        return image, self.images[idx]  # 返回图像及其文件名
