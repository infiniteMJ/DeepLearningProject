import os
import os.path
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image

class DerainDataset(Dataset):
    def __init__(self, data_root, ground_truth, rainy, transform=None):
        super(DerainDataset, self).__init__()
        self.ground_truth = ground_truth
        self.rainy = rainy
        self.gt_path = os.path.join(data_root, ground_truth)
        self.rainy_path = os.path.join(data_root, rainy)
        self.gt_files = os.listdir(self.gt_path)
        self.rainy_files = os.listdir(self.rainy_path)
        self.transform = transform

    def __len__(self):
        return len(self.rainy_files)

    def __getitem__(self, idx):
        gt_img = Image.open(os.path.join(self.gt_path, str(idx) + '.jpg')).convert('RGB')
        if self.transform:
            gt_img = self.transform(gt_img)
        else:
            gt_img = transforms.functional.to_tensor(gt_img)

        rainy_img = []
        for i in range(1, 15):
            rain_img = Image.open(os.path.join(self.rainy_path, str(idx) + '_' + str(i) + '.jpg')).convert('RGB')
            if self.transform:
                rain_img = self.transform(rain_img)
            else:
                rain_img = transforms.functional.to_tensor(rain_img)
            rainy_img.append(rain_img)

        sample = {
            "ground_truth": gt_img,
            'rainy_img': rainy_img
        }
        return sample


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((360, 240)),
        transforms.ToTensor()
    ])

    train_dataset = DerainDataset(data_root='drive/MyDrive/DeepLearningProject/data/training',
                                  ground_truth="ground_truth",
                                  rainy='rainy_image',
                                  transform=transform)
