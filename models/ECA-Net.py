import torch
import torch.nn as nn

class ECA(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c, 1)
        y = self.conv(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y

class ECANet(nn.Module):
    def __init__(self, num_classes=3, input_channels=3):
        super(ECANet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.eca = ECA(64)
        self.conv2 = nn.Conv2d(64, num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = self.eca(x)
        return self.conv2(x)
