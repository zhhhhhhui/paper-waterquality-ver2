import torch
import torch.nn as nn
from layers.capsule import CapsuleLayer
from layers.capsule_conv_layer import CapsuleConvLayer
from layers.mdia import MDIA


class HybridNet(nn.Module):
    def __init__(self, image_width, image_height, image_channels, conv_input_channel,
                 conv_output_channel, num_primary_node, primary_node_size, num_output_node, output_node_size, heads=8):
        super(HybridNet, self).__init__()

        # MDIA
        self.mdia = MDIA(input_dim=image_channels, heads=heads)

        self.capsule_conv = CapsuleConvLayer(in_channels=conv_input_channel, out_channels=conv_output_channel)

        self.primary_capsule = CapsuleLayer(
            in_node_num=num_primary_node,
            in_size=conv_output_channel,
            num_node=num_primary_node,
            node_size=primary_node_size,
            use_routing=False
        )

        self.digits_capsule = CapsuleLayer(
            in_node_num=num_primary_node,
            in_size=primary_node_size,
            num_node=num_output_node,
            node_size=output_node_size,
            use_routing=True
        )

    def forward(self, x):
        x = self.mdia(x)

        # Reshape
        x = x.view(x.size(0), 1, x.size(1), x.size(2), x.size(3))

        x = self.capsule_conv(x)

        x = self.primary_capsule(x)

        output = self.digits_capsule(x)

        return output
