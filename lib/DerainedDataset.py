import os
import os.path
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from PIL import Image
from skimage import io
from math import ceil

class DerainDataset(Dataset):
    def __init__(self, data_root, ground_truth, rainy, patch_size=0, transform=None):
        super(DerainDataset, self).__init__()
        self.ground_truth = ground_truth
        self.rainy = rainy
        self.gt_path = os.path.join(data_root, ground_truth)
        self.rainy_path = os.path.join(data_root, rainy)
        self.rainy_files = os.listdir(self.rainy_path)
        self.patch_size = patch_size
        self.transform = transform

    def __len__(self):
        return len(self.rainy_files)

    def __getitem__(self, key):
        new_key = key + 1
        idx = ceil(new_key / 14)
        gt_img = Image.open(os.path.join(self.gt_path, str(idx) + '.jpg')).convert('RGB')
        sub = new_key % 14 if new_key % 14 !=0 else 14
        rain_img = Image.open(os.path.join(self.rainy_path, str(idx) + '_' + str(sub) + '.jpg')).convert('RGB')

        if self.transform:
            gt_img = self.transform(gt_img)
            rain_img = self.transform(rain_img)
        else:
            gt_img = transforms.functional.to_tensor(gt_img)
            rain_img = transforms.functional.to_tensor(rain_img)

        if self.patch_size:
            i, j, h, w = transforms.RandomCrop.get_params(
                gt_img, output_size=(self.patch_size, self.patch_size))
            gt_img = TF.crop(gt_img, i, j, h, w)
            rain_img = TF.crop(rain_img, i, j, h, w)

        sample = {
            "ground_truth": gt_img,
            'rain_img': rain_img
        }

        return sample
