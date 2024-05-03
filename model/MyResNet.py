import torch
import torch.nn as nn
from torchvision.models import resnet50


class MyResNet(nn.Module):
    def __init__(self):
        super(MyResNet, self).__init__()
        model = resnet50(weights='ResNet50_Weights.DEFAULT')
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 20)
        self.resnet = model

    def forward(self, x):
        x = self.resnet(x)
        return x