import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn.functional as F


class ConvUnit(nn.Module):
    def __init__(self, in_channels):
        super(ConvUnit, self).__init__()
        # self.conv0 = nn.Conv2d(in_channels=in_channels, #256
        #                        out_channels=32,
        #                        kernel_size=9,
        #                        stride=2,
        #                        bias=True)
        self.conv0 = nn.Conv2d(in_channels=in_channels,
                               out_channels=32,
                               kernel_size=9,
                               stride=2,
                               bias=True)
        # [24,32,22,22]

    def forward(self, x):
        return self.conv0(x)


class CapsuleLayer(nn.Module):
    def __init__(self, in_node_num, in_size, num_node, node_size, use_routing):
        super(CapsuleLayer, self).__init__()

        self.in_node_num = in_node_num
        self.in_size = in_size
        self.num_node = num_node
        self.use_routing = use_routing
        self.node_size = node_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.use_routing:

            self.W = nn.Parameter(torch.randn(1, in_node_num, num_node, node_size, in_size))  # *0.03

            # W.shape(1,32*6*6,10,16,8) why 1 here

        else:

            def create_conv_unit(unit_idx):
                unit = ConvUnit(in_channels=in_size)
                self.add_module('unit_' + str(unit_idx), unit)
                return unit

            self.units = [create_conv_unit(i) for i in range(node_size)]

    @staticmethod
    def squash(s):
        # s.shape ? (batch,node_size,num)
        # print(s.type())
        mag_sq = torch.sum(s ** 2, dim=1, keepdim=True)
        # print(s.size())
        mag = torch.sqrt(mag_sq)
        s = (mag_sq / (1.0 + mag_sq)) * (s / mag)
        return s

    def forward(self, x):
        if self.use_routing:
            return self.routing(x)
        else:
            return self.no_routing(x)

    def no_routing(self, x):
        # x(batch,channel256,height,weight)
        u = [self.units[i](x) for i in range(self.node_size)]
        u = torch.stack(u, dim=1)
        # u(batch,in_size8,channel32,height,weight)
        # (128,8,32,6,6)
        # flatten(batch,unit_size8,-1)
        u = u.view(x.size(0), self.node_size, -1)
        # u(128,8,32*6*6)

        return CapsuleLayer.squash(u)

    def routing(self, x):
        # x(batch_size, node_size, in_size * height * width)
        # (128,8,32*6*6)
        batch_size = x.size(0)
        x = x.transpose(1, 2)

        # x = x.transpose(1, 2)
        x = torch.stack([x] * self.num_node, dim=2).unsqueeze(4)

        # (batch,num_nodes,out_num_nodes,node_size8,1)
        # (batch_size, in_size * height * width, self.num_node, node_size, 1)
        # (batch,num_nodes,out_num_nodes,node_size8,1)

        W = torch.cat([self.W] * batch_size, dim=0) * 0.03
        # (batch,num_nodes,out_num_nodes,out_node_size16,node_size8)

        u_hat = torch.matmul(W, x)
        # test0 = torch.sum(u_hat ** 2, dim=3)

        # (batch,num_nodes,out_num_nodes,out_node_size16,1)

        b_ij = torch.zeros(1, self.in_node_num, self.num_node, 1).to(self.device)
        num_iterations = 3
        for iteration in range(num_iterations):
            # print(b_ij.size())
            c_ij = F.softmax(b_ij, dim=2)
            # if iteration==0:
            # c_ij[:,:,:,:]=0.0009
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)

            # c_ij(128,32*6*6,10,1,1)

            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)

            v_j = CapsuleLayer.squash(s_j.squeeze().transpose(1, 2))

            # test=torch.sum(v_j**2,dim=1)
            v_j = v_j.transpose(1, 2)[:, None, :, :, None]
            v_j1 = torch.cat([v_j] * self.in_node_num, dim=1)

            # print(v_j1.size())
            # (batch,num_nodes,out_num_nodes,out_node_size,1)
            # (128,32*6*6,10,16,1)
            # u_hat(128,32*6*6,10,16,1)->(128,32*6*6,10,1,16)
            u_vj1 = torch.matmul(u_hat.transpose(3, 4), v_j1).squeeze(4).mean(dim=0, keepdim=True)

            b_ij = b_ij + u_vj1

        # test = torch.sum(v_j.squeeze() ** 2, dim=2)
        return v_j.squeeze(1)
