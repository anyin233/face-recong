import torch
import torch.nn as nn
import torch.optim as optim


# identity block doesn't change input sizes and the number of channels. at least in this course.
class identity_block(nn.Module):
    def __init__(self, filters, in_channels):
        super(identity_block, self).__init__()
        F1, F2 = filters
        # First component of the main path
        self.conv2d_1 = nn.Conv2d(in_channels, F1, kernel_size=1)  # doesn't change input size
        self.bn_1 = nn.BatchNorm2d(F1)
        self.relu_1 = nn.ReLU()

        # Second component of the main path
        self.conv2d_2 = nn.Conv2d(F1, F2, kernel_size=3, padding=1)  # same convolution
        self.bn_2 = nn.BatchNorm2d(F2)
        self.relu_2 = nn.ReLU()

        # Third component of the main path
        # we get back to the original channel size since we assume that the input from the shortcut path and
        # from the main path has the same number of channels when we are using an identity block.
        self.conv2d_3 = nn.Conv2d(F2, in_channels, kernel_size=1)  # doesn't change input size
        self.bn_3 = nn.BatchNorm2d(in_channels)
        self.relu_3 = nn.ReLU()

    def forward(self, x):
        x_shortcut = x

        x = self.conv2d_1(x)
        x = self.bn_1(x)
        x = self.relu_1(x)

        x = self.conv2d_2(x)
        x = self.bn_2(x)
        x = self.relu_2(x)

        x = self.conv2d_3(x)
        x = self.bn_3(x)

        x += x_shortcut

        x = self.relu_3(x)

        return x


class convolutional_block(nn.Module):
    def __init__(self, filters, s, in_channels):
        super(convolutional_block, self).__init__()

        F1, F2, F3 = filters

        # First component of the main path
        self.conv2d_1 = nn.Conv2d(in_channels, F1, kernel_size=1, stride=s)
        self.bn_1 = nn.BatchNorm2d(F1)
        self.relu_1 = nn.ReLU()

        # Second component of the main path
        self.conv2d_2 = nn.Conv2d(F1, F2, kernel_size=3, padding=1)  # 3x3 same convolution
        self.bn_2 = nn.BatchNorm2d(F2)
        self.relu_2 = nn.ReLU()

        # Third component of the main path
        self.conv2d_3 = nn.Conv2d(F2, F3, kernel_size=1)
        self.bn_3 = nn.BatchNorm2d(F3)

        # Shortcut path
        # After this, input sizes and the number of channels will match. Because this convolution applies the only
        # transformation that affects the input size, the key point here is stride=s. It's output channels is equal
        # to the number of output channels of the main path.
        self.conv2d_shortcut = nn.Conv2d(in_channels, F3, kernel_size=1, stride=s)
        self.bn_shortcut = nn.BatchNorm2d(F3)

        self.relu_3 = nn.ReLU()

    def forward(self, x):
        x_shortcut = x

        x = self.conv2d_1(x)
        x = self.bn_1(x)
        x = self.relu_1(x)

        x = self.conv2d_2(x)
        x = self.bn_2(x)
        x = self.relu_2(x)

        x = self.conv2d_3(x)
        x = self.bn_3(x)

        x_shortcut = self.conv2d_shortcut(x_shortcut)
        x_shortcut = self.bn_shortcut(x_shortcut)

        x += x_shortcut
        x = self.relu_3(x)

        return x


class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size()
        return x.view(N, -1)


class ResNet50(nn.Module):
    def __init__(self, num_classes, feature=False):
        super().__init__()

        self.padding = nn.ConstantPad2d(3, 0)

        self.conv0 = nn.Conv2d(3, 64, kernel_size=7, stride=2)
        self.bn0 = nn.BatchNorm2d(64)
        self.mp0 = nn.MaxPool2d(kernel_size=3, stride=2)

        # Stage 2
        self.cb0 = convolutional_block([64, 64, 256], s=1, in_channels=64)
        self.ib0 = identity_block([64, 64], in_channels=256)
        self.ib1 = identity_block([64, 64], in_channels=256)
        # Stage 3
        self.block0 = nn.Sequential(
            convolutional_block([128, 128, 512], s=2, in_channels=256),
            identity_block([128, 128], in_channels=512),
            identity_block([128, 128], in_channels=512),
            identity_block([128, 128], in_channels=512)
        )
        # Stage 4
        self.block1 = nn.Sequential(
            convolutional_block([256, 256, 1024], s=2, in_channels=512),
            identity_block([256, 256], in_channels=1024),
            identity_block([256, 256], in_channels=1024),
            identity_block([256, 256], in_channels=1024),
            identity_block([256, 256], in_channels=1024),
            identity_block([256, 256], in_channels=1024)
        )
        # Stage 5
        self.block2 = nn.Sequential(
            convolutional_block([512, 512, 2048], s=2, in_channels=1024),
            identity_block([512, 512], in_channels=2048),
            identity_block([512, 512], in_channels=2048),
        )
        # ---
        self.ap0 = nn.AvgPool2d(kernel_size=2)  # outputs 1x1x2048
        self.flatten = Flatten()
        self.feature = feature
        if not feature:
            self.fc0 = nn.Linear(8192, 2048)
            self.relu0 = nn.ReLU()
            self.output = nn.Linear(2048, num_classes)
            self.output_act = nn.Sigmoid()

    def forward(self, x):
        x = self.padding(x)

        x = self.conv0(x)
        x = self.bn0(x)
        x = self.mp0(x)

        x = self.cb0(x)
        x = self.ib0(x)
        x = self.ib1(x)

        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)

        x = self.ap0(x)
        flatten = self.flatten(x)
        if not self.feature:
            flatten = self.fc0(flatten)
            flatten = self.relu0(flatten)
            return self.output_act(self.output(flatten))
        else:
            return flatten
