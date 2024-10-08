import torch
import torch.nn as nn
import torch.nn.functional as F


class SWUBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SWUBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.swish = nn.SiLU()  # SWISH activation

    def forward(self, x):
        x = self.swish(self.conv1(x))
        return self.conv2(x)


class SWUNet(nn.Module):
    def __init__(self, num_classes=3):
        super(SWUNet, self).__init__()
        self.encoder = nn.Sequential(
            SWUBlock(3, 64),
            SWUBlock(64, 128)
        )
        self.decoder = nn.Sequential(
            SWUBlock(128, 64),
            SWUBlock(64, 32)
        )
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return self.final_conv(x)
