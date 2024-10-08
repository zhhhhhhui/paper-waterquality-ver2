import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
#胶囊卷积层
'''
定义了一个胶囊卷积层，包括了卷积操作和激活函数的处理过程。

它可以被用作胶囊网络中的一个构建块。在前向传播过程中，输入数据经过卷积操作和 ReLU 激活函数，得到处理后的输出
'''
# class CapsuleConvLayer(nn.Module):
#     def __init__(self,in_channels,out_channels):
#         super(CapsuleConvLayer,self).__init__()
#
#         self.conv0=nn.Conv2d(in_channels=in_channels, #
#                              out_channels=out_channels,#
#                              kernel_size=9,
#                              stride=1,
#                              bias=True)
#         self.relu=nn.ReLU(inplace=True)
#
#     def forward(self,x):
#         x=self.conv0(x)
#         x=self.relu(x)
#         # print(x.shape)
#
#
#         # [24, 1024, 56, 56]
#         return x
         # [样本数 ，1，30，28，28]
class CapsuleConvLayer(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(CapsuleConvLayer, self).__init__()
        # [24,1,30,28,28]
        self.conv3d1 = nn.Conv3d(in_channels=in_channels, out_channels=32, kernel_size=(9,7,7))
        # [24,32,24,22,22]
        self.relu1 = nn.ReLU()
        self.maxpool3d1 = nn.MaxPool3d(kernel_size=(5,3,3), stride=1)
        # [24,32,18,20,20]
        # self.conv3d2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3)
        # self.relu2 = nn.ReLU()
        # self.maxpool3d2 = nn.MaxPool3d(kernel_size=2, stride=2)
        ## [24, 64, 14, 14, 42]
        # W（out） = (W（in） + 2P（填充） -W(核））/S（步幅） + 1
        # 目标[128 * 52 * 52]
        # [样本数，576，20，20]



    def forward(self, x):
        # 3D Con部分
        x3d = self.conv3d1(x.float())
        # [24, 64, 56, 56, 150]
        x3d = self.relu1(x3d)
        x3d = self.maxpool3d1(x3d)
        # # [24, 64, 24, 24, 64]
        # x3d = self.conv3d2(x3d)
        # # # [24, 64, 29, 29, 85]
        # x3d = self.relu2(x3d)
        # # print(x3d.shape)
        # x3d = self.maxpool3d2(x3d)
        # print(x3d.shape)
        # # [24, 64, 14, 14, 42]

        x = x3d.reshape(x3d.shape[0],-1,x3d.shape[-2],x3d.shape[-1])
        # print(x3d.shape)
        # x3d = self.conv2d3(x3d)
        # x = self.relu3(x3d)

        return x