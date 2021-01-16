import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=(3, 3), stride=(1, 1)):
        super().__init__()
        self.conv0 = nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=kernel_size,
            stride=stride
        )
        self.bn0 = nn.BatchNorm2d(out_channel)
        self.act0 = nn.ReLU()

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        return self.act0(x)


class ConvBlock2Layers(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=(3, 3), stride=(1, 1), pool_kernel=(2, 2),
                 pool_stride=(2, 2)):
        super().__init__()
        self.block0 = ConvBlock(in_channel, out_channel, kernel_size, stride)
        self.block1 = ConvBlock(out_channel, out_channel, kernel_size, stride)
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride)

    def forward(self, x):
        x = self.block0(x)
        x = self.block1(x)
        return self.pool(x)


class ConvBlock3Layers(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=(3, 3), stride=(1, 1), pool_kernel=(2, 2),
                 pool_stride=(2, 2)):
        super().__init__()
        self.block0 = ConvBlock(in_channel, out_channel, kernel_size, stride)
        self.block1 = ConvBlock(out_channel, out_channel, kernel_size, stride)
        self.block2 = ConvBlock(out_channel, out_channel, kernel_size, stride)
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride)

    def forward(self, x):
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        return self.pool(x)


class FCBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.5, activation=nn.ReLU()):
        super().__init__()
        self.fc1 = nn.Linear(
            in_features=in_features,
            out_features=out_features
        )
        self.bn = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(p=dropout)
        self.act = activation

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn(x)
        x = self.dropout(x)
        return self.act(x)


class Vgg16(nn.Module):
    def __init__(self, num_class, feature=False):
        super().__init__()
        self.num_class = num_class
        self.conv0 = ConvBlock2Layers(3, 64)
        self.conv1 = ConvBlock2Layers(64, 128)
        self.conv2 = ConvBlock3Layers(128, 256)
        self.conv3 = ConvBlock3Layers(256, 512)
        self.flatten = nn.Flatten()
        self.feature = feature
        if not feature:
            self.fc0 = FCBlock(2048, 2048)
            self.fc1 = FCBlock(2048, 1024)
            self.output = nn.Linear(1024, num_class)
            self.output_act = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        if not self.feature:
            x = self.fc0(x)
            x = self.fc1(x)
            x = self.output(x)
            x = self.output_act(x)
        return x
