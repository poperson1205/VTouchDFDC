# https://github.com/fangpin/siamese-pytorch/blob/master/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

class Siamese(nn.Module):

    def __init__(self):
        super(Siamese, self).__init__()
        self.efficient_net = EfficientNet.from_pretrained('efficientnet-b0', num_classes=1)
        self.linear = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Sigmoid())
        self.out = nn.Linear(1280, 1)

    def forward_one(self, x):
        x = self.efficient_net.extract_features(x)
        x = self.linear(x)
        x = x.view(x.size()[0], -1)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        dis = torch.abs(out1 - out2)
        out = self.out(dis)
        #  return self.sigmoid(out)
        return out