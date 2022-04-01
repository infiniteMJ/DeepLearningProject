import torch
import torch.nn as nn
import torch.nn.functional as F
from guided_filter_pytorch.guided_filter import GuidedFilter

class DerainNet(nn.Module):
    def __init__(self):
        super(DerainNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 512, (3, 3), padding=2)
        self.conv2 = nn.Conv2d(512, 16, (3, 3), padding=1)
        self.conv3 = nn.Conv2d(16, 3, (3, 3), padding=0)

    def forward(self, image):
        base = GuidedFilter(15, 1)(image, image)
        detail = image - base
        
        y = self.conv1(detail)
        y = F.relu(y)
        y = self.conv2(y)
        y = F.relu(y)
        y = self.conv3(y)

        return y
