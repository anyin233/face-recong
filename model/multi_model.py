import torch.nn as nn
import torch
from model.ResNet import ResNet50
from model.vgg16 import Vgg16


class ConcatModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet = ResNet50(num_classes, feature=True)
        self.vgg = Vgg16(num_classes, feature=True)
        self.fc0 = nn.Linear(8192 + 2048, 2048)
        self.act0 = nn.ReLU()
        self.fc1 = nn.Linear(2048, 1024)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 512)
        self.act2 = nn.ReLU()
        self.output = nn.Linear(512, num_classes)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        resnet = self.resnet(x)
        vgg = self.vgg(x)
        feature = torch.cat([resnet, vgg], dim=1)
        feature = self.fc0(feature)
        feature = self.act0(feature)
        feature = self.fc1(feature)
        feature = self.act1(feature)
        feature = self.fc2(feature)
        feature = self.act2(feature)
        feature = self.output(feature)
        return self.output_act(feature)
