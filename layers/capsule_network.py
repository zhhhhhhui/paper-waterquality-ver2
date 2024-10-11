import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.utils as utils
import torch.nn.functional as F
from capsule_conv_layer import CapsuleConvLayer
from capsule_layer import CapsuleLayer


class CapsuleNetwork(nn.Module):
    def __init__(self,
                 image_width,
                 image_height,
                 image_channels,
                 conv_input_channel,

                 conv_output_channel,
                 num_primary_node,
                 primary_node_size,
                 num_output_node,
                 output_node_size):
        super(CapsuleNetwork, self).__init__()
        self.reconstructed_image_count = 0
        self.image_channels = image_channels
        self.image_width = image_width
        self.image_height = image_height

        self.conv1 = CapsuleConvLayer(in_channels=1,  # conv_input_channel = 176
                                      out_channels=conv_output_channel,  # conv_output_channel = 512
                                      )
        self.primary = CapsuleLayer(in_node_num=0,
                                    in_size=conv_output_channel,  # 512
                                    num_node=num_primary_node,  # 32 * 6 * 6
                                    node_size=primary_node_size,  # 126
                                    use_routing=False)
        self.digits = CapsuleLayer(in_node_num=num_primary_node,  # 32 * 6 * 6
                                   in_size=primary_node_size,  # 8
                                   num_node=num_output_node,  # 10
                                   node_size=output_node_size,  # 16
                                   use_routing=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # reconstruction_size = image_width * image_height * image_channels  # masked

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.digits(self.primary(self.conv1(x)))

    def loss(self, images, input, target, size_average=True):
        return self.margin_loss(input, target, size_average)
        # + self.reconstruction_loss(images, input, size_average)

    def margin_loss(self, input, target, size_average=True):
        batch_size = input.size(0)
        v_mag = torch.sqrt((input ** 2).sum(dim=2, keepdim=True))
        # (128, 10, 16, 1)
        # (batch,num_nodes,1,1?)
        zero = Variable(torch.zeros(1)).to(self.device)
        m_plus = 0.9
        m_minus = 0.1
        max_l = torch.max(m_plus - v_mag, zero).view(batch_size, -1) ** 2
        max_r = torch.max(v_mag - m_minus, zero).view(batch_size, -1) ** 2
        loss_lambda = 0.5
        T_c = target
        # print(target.size()) 128,10
        L_c = T_c * max_l + loss_lambda * (1 - T_c) * max_r
        L_c = L_c.sum(dim=1)
        if size_average:
            L_c = L_c.mean()
        return L_c
