from DerainedDataset import DerainDataset
from train_derain import train_model
from DerainNet import DerainNet
import os
import os.path
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from guided_filter_pytorch.guided_filter import GuidedFilter
import cv2
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from PIL import Image
from skimage import io
from math import ceil

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()
    model_CNN = DerainNet().to(device)
    optimizer = optim.Adam(model_CNN.parameters(), lr=0.01)
    model = train_model(model_CNN, criterion, optimizer)
