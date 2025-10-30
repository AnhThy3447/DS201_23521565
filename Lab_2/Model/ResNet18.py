import torch
from torch import nn
from torch.nn import functional as F

class ResNet_Block(nn.Module):
    def __init__(self, in_channel, out_channel, strides=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=3,
            padding=1,
            stride=strides
        )

        self.conv2 = nn.Conv2d(
            in_channels=out_channel,
            out_channels=out_channel,
            kernel_size=3,
            padding=1,
            stride=strides
        )

        if in_channel != out_channel:
            self.conv3 = nn.Conv2d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=1,
                stride=strides
            )
        else:
            self.conv3 = None
        
        self.bn = nn.BatchNorm2d(out_channel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.bn(self.conv1(x))
        out = F.relu(out)
        out = self.bn(self.conv2(out))
        if self.conv3():
            x = self.conv3(x)
        out += x
        return F.relu(out)
    

class ResNet18(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super(ResNet18, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            padding=3,
            stride=2
        )

        self.bn = nn.BatchNorm2d(64)

        self.Maxpool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=0 ######
        )

        self.layer1 = self.create_layer(64, 2, stride=1)
        self.layer2 = self.create_layer(128, 2, stride=2)
        self.layer3 = self.create_layer(256, 2, stride=2)
        self.layer4 = self.create_layer(512, 2, stride=2)

        self.Avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, num_classes)

    def create_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResNet_Block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn(self.conv(x)))
        x = self.Maxpool(x)

        x = self.Maxpool(self.layer1(x))
        x = self.Maxpool(self.layer2(x))
        x = self.Maxpool(self.layer3(x))
        x = self.Maxpool(self.layer4(x))

        x = self.Avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x