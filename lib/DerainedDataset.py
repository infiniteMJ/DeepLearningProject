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
from skimage import io

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
        gt_img = io.imread(os.path.join(self.gt_path, str(idx) + '.jpg'))
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGRA2BGR)
        gt_img = transforms.functional.to_tensor(gt_img)
        if self.transform:
            gt_img = self.transform(gt_img)

        rainy_img = []
        for i in range(1, 15):
            rain_img = io.imread(os.path.join(self.rainy_path, str(idx) + '_' + str(i) + '.jpg'))
            rain_img = cv2.cvtColor(rain_img, cv2.COLOR_BGRA2BGR)
            rain_img = transforms.functional.to_tensor(rain_img)
            if self.transform:
                rain_img = self.transform(rain_img)
            rainy_img.append(rain_img)

        sample = {
            "ground_truth": gt_img,
            'rainy_img': rainy_img
        }
        return sample