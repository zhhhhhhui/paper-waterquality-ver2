import torch
import torch.nn as nn

class CASBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CASBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = self.skip(x)
        x = nn.ReLU()(self.conv1(x))
        x = self.conv2(x)
        return nn.ReLU()(x + residual)

class CASNet(nn.Module):
    def __init__(self, num_classes=3):
        super(CASNet, self).__init__()
        self.encoder = nn.Sequential(
            CASBlock(3, 64),
            CASBlock(64, 128)
        )
        self.decoder = nn.Sequential(
            CASBlock(128, 64),
            CASBlock(64, 32)
        )
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return self.final_conv(x)
