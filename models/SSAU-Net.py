import torch
import torch.nn as nn

class SAUBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SAUBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.attention = nn.Sequential(
            nn.Conv2d(out_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        residual = x
        x = nn.ReLU()(self.conv1(x))
        x = self.conv2(x)
        attn = self.attention(x)
        return nn.ReLU()(x + attn * residual)

class SAUNet(nn.Module):
    def __init__(self, num_classes=3):
        super(SAUNet, self).__init__()
        self.encoder = nn.Sequential(
            SAUBlock(3, 64),
            SAUBlock(64, 128)
        )
        self.decoder = nn.Sequential(
            SAUBlock(128, 64),
            SAUBlock(64, 32)
        )
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return self.final_conv(x)
