from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
from PIL import Image
from unet import UNet  # 假设 UNet 是 unet.py 中定义的一个类
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
def iou(preds, labels):
    # print(preds)
    # print*(labels)
    # 计算交集
    intersection = (preds & labels).float().sum()
    if intersection.item() == 0:
        # 如果交集为0，立即返回0
        return 0.0

    # 计算并集
    union = (preds | labels).float().sum()

    # 计算IoU
    iou_score = intersection / union
    return iou_score

class KITTIDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        return image, mask

transform = transforms.Compose([
     transforms.Resize((256, 1024)),
    transforms.ToTensor()
    
])

dataset = KITTIDataset('data_semantics/training/image_2', 'data_semantics/training/semantic', transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet( ).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

def train(model, loader, optimizer, criterion, n_epochs):
    for epoch in range(n_epochs):
        model.train()
        for i, (images, masks) in enumerate(loader):
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            # if i % (len(loader) - 1) == 0:  # 最后一个batch
            #     img = images[0].cpu().detach().numpy().transpose(1, 2, 0)
            #     pred = outputs[0].cpu().detach().numpy().argmax(axis=0)  # 假设输出是类别概率
            #     true_mask = masks[0].cpu().numpy()

            #     plt.figure(figsize=(10, 5))
            #     plt.subplot(1, 3, 1)
            #     plt.imshow(img, cmap='gray')
            #     plt.title('Original Image')
            #     plt.subplot(1, 3, 2)
            #     plt.imshow(pred, cmap='gray')
            #     plt.title('Predicted Mask')
            #     plt.subplot(1, 3, 3)
            #     true_mask=true_mask.squeeze(0)
            #     plt.imshow(true_mask, cmap='gray')
            #     plt.title('True Mask')
            #     plt.show()
        
        iou_score = iou((outputs > 0.5).int(), masks.int())
        print(f'Epoch {epoch+1}, Loss: {loss.item()}, IoU: {iou_score}')
train(model, dataloader, optimizer, criterion, 10)
