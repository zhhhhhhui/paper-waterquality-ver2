import torch
import torch.nn as nn
import torch.nn.functional as F


class CapsuleLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_capsules, num_routing=3):
        super(CapsuleLayer, self).__init__()
        self.num_capsules = num_capsules
        self.num_routing = num_routing
        self.W = nn.Parameter(torch.randn(1, in_dim, num_capsules * out_dim))

    def forward(self, x):
        batch_size = x.size(0)
        x = torch.matmul(x, self.W)
        x = x.view(batch_size, self.num_capsules, -1)
        for _ in range(self.num_routing):
            c = F.softmax(x, dim=1)
            x = (c.unsqueeze(2) * x.unsqueeze(1)).sum(dim=1)
        return x
