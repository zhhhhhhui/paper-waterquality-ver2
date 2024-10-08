import torch
import torch.nn as nn

class GLCSA(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(GLCSA, self).__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)

    def forward(self, x):
        w = torch.mean(x, dim=(2, 3), keepdim=True)
        w = self.fc1(w)
        w = nn.ReLU()(w)
        w = self.fc2(w)
        w = torch.sigmoid(w)
        return x * w

class GLSCANet(nn.Module):
    def __init__(self, num_classes=3, input_channels=3):
        super(GLSCANet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.glsca = GLCSA(64)
        self.conv2 = nn.Conv2d(64, num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = self.glsca(x)
        return self.conv2(x)
