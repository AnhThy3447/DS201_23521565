import torch
from torch import nn
from torch.nn import functional as F

class LeNet(nn.Module):
   def __init__(self, num_classes: int) -> None:
      super(LeNet, self).__init__()

      self.conv1 = nn.Conv2d(
         in_channels=1,
         out_channels=6,
         kernel_size=5,
         padding=2
      )

      self.pool = nn.AvgPool2d(
         kernel_size=2,
         stride=2,
         padding=0
      )

      self.conv2 = nn.Conv2d(
         in_channels=6,
         out_channels=16,
         kernel_size=5,
         padding=0
      )

      self.fc1 = nn.Linear(16*5*5, 120)
      self.fc2 = nn.Linear(120, 84)
      self.fc3 = nn.Linear(84, num_classes)
   
   def forward(self, x: torch.Tensor) -> torch.Tensor:
      x = F.sigmoid(self.conv1(x))
      x = self.pool(x)

      x = F.sigmoid(self.conv2(x))
      x = self.pool(x)

      x = torch.flatten(x, 1)

      x = F.sigmoid(self.fc1(x))
      x = F.sigmoid(self.fc2(x))
      x = self.fc3(x)
      return x


