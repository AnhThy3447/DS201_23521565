import torch
from torch import nn
from torch.nn import functional as F

class InceptionBlocks(nn.Module):
    def __init__(self, c1: int, c2: int, c3: int, c4: int) -> None:
        super(InceptionBlocks, self).__init__()

        # Branch 1
        self.b1_1 = nn.LazyConv2d(
            out_channels=c1,
            kernel_size=1
        )

        # Branch 2
        self.b2_1 = nn.LazyConv2d(
            out_channels=c2[0],
            kernel_size=1
        )

        self.b2_2 = nn.LazyConv2d(
            out_channels=c2[1],
            kernel_size=3,
            padding=1
        )

        # Branch 3
        self.b3_1 = nn.LazyConv2d(
            out_channels=c3[0],
            kernel_size=1
        )

        self.b3_2 = nn.LazyConv2d(
            out_channels=c3[1],
            kernel_size=5,
            padding=2
        )

        # Branch 4
        self.b4_1 = nn.MaxPool2d(
            kernel_size=3,
            stride=1,
            padding=1,
            ceil_mode=True
        )
      
        self.b4_2 = nn.LazyConv2d(
            out_channels=c4,
            kernel_size=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = F.relu(self.b1_1(x))
        b2 = F.relu(self.b2_2(F.relu(self.b2_1(x))))
        b3 = F.relu(self.b3_2(F.relu(self.b3_1(x))))
        b4 = F.relu(self.b4_2(self.b4_1(x)))
        return torch.cat((b1, b2, b3, b4), dim=1)
    
class GoogLeNet(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super(GoogLeNet, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3
        )

        self.conv2 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=1
        )

        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=192,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.Maxpool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1,
            ceil_mode=True
        )

        self.Avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

        # --- Inception khối 3 ---
        self.inception3a = InceptionBlocks(64, (96, 128), (16, 32), 32)
        self.inception3b = InceptionBlocks(128, (128, 192), (32, 96), 64)

        # --- Inception khối 4 ---
        self.inception4a = InceptionBlocks(192, (96, 208), (16, 48), 64)
        self.inception4b = InceptionBlocks(160, (112, 224), (24, 64), 64)
        self.inception4c = InceptionBlocks(128, (128, 256), (24, 64), 64)
        self.inception4d = InceptionBlocks(112, (144, 288), (32, 64), 64)
        self.inception4e = InceptionBlocks(256, (160, 320), (32, 128), 128)

        # --- Inception khối 5 ---
        self.inception5a = InceptionBlocks(256, (160, 320), (32, 128), 128)
        self.inception5b = InceptionBlocks(384, (192, 384), (48, 128), 128)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.Maxpool(x)

        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.Maxpool(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.Maxpool(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.Maxpool(x)

        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.Avgpool(x)

        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x