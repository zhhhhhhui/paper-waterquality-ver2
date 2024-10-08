import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.utils as utils
import torch.nn.functional as F
from capsule_conv_layer import CapsuleConvLayer
from capsule_layer import CapsuleLayer

'''
实现了一个名为 CapsuleNetwork 的胶囊网络模型
'''
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
                                      out_channels=conv_output_channel, #conv_output_channel = 512
                                      )
        self.primary = CapsuleLayer(in_node_num=0,
                                    in_size=conv_output_channel, #512
                                    num_node=num_primary_node, #32 * 6 * 6
                                    node_size=primary_node_size,#126
                                    use_routing=False)
        self.digits = CapsuleLayer(in_node_num=num_primary_node,#32 * 6 * 6
                                   in_size=primary_node_size, #8
                                   num_node=num_output_node, #10
                                   node_size=output_node_size,#16
                                   use_routing=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # reconstruction_size = image_width * image_height * image_channels  # masked
        # '''
        # nn.Linear 是 PyTorch 中的一个类，用于定义线性变换（全连接层）。它是神经网络中常用的一种基本操作，用于将输入数据线性映射到输出空间
        # '''
        #
        # self.reconstruct0 = nn.Linear(num_output_node * output_node_size, int((reconstruction_size * 2) / 3))
        # self.reconstruct1 = nn.Linear(int((reconstruction_size * 2) / 3), int((reconstruction_size * 3) / 2))
        # self.reconstruct2 = nn.Linear(int((reconstruction_size * 3) / 2), reconstruction_size)
        """
        self.reconstruct0 = nn.Linear(num_output_node * output_node_size, int((reconstruction_size * 2) / 3))：
        这是第一个线性全连接层，将输出胶囊向量的表示映射到一个更低维度的表示空间，以便进行更紧凑的特征表示。
        num_output_node * output_node_size 是输出胶囊向量的大小。
        reconstruction_size * 2 / 3 表示第一次映射后的表示空间维度，这是一个较小的维度。
        这个线性层的作用是进行特征压缩和提取。
        
        self.reconstruct1 = nn.Linear(int((reconstruction_size * 2) / 3), int((reconstruction_size * 3) / 2))：
        这是第二个线性全连接层，将第一个映射后的表示空间进一步映射到更高维度的表示空间，以便更好地捕获图像特征。
        int((reconstruction_size * 2) / 3) 是第一个映射后的表示空间维度。
        reconstruction_size * 3 / 2 表示第二次映射后的表示空间维度，这是一个较大的维度。
        self.reconstruct2 = nn.Linear(int((reconstruction_size * 3) / 2), reconstruction_size)：

        这是第三个线性全连接层，将第二个映射后的表示空间映射回与输入图像像素数量相等的表示空间，从而实现最终的图像重建。
        int((reconstruction_size * 3) / 2) 是第二个映射后的表示空间维度。
        reconstruction_size 是输入图像的像素数量
        """

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.digits(self.primary(self.conv1(x)))
    '''
    在深度学习模型的前向传播过程中，各个层的输出作为输入传递给下一个层，从而构建整个计算图
    '''

    def loss(self, images, input, target, size_average=True):
        return self.margin_loss(input, target, size_average)
               # + self.reconstruction_loss(images, input, size_average)

    #这里的input相当于network(data)，也就是输出的结果
    '''
    这段代码计算了一个损失函数，称为Margin Loss，通常用于胶囊网络中。它的目的是确保每个输出胶囊在输入样本的特征空间中有足够的间隔，以便能够更好地分类不同的输出类别
    '''
    def margin_loss(self, input, target, size_average=True):

        batch_size = input.size(0)
        v_mag = torch.sqrt((input ** 2).sum(dim=2, keepdim=True))
        '''
        v_mag：计算每个胶囊输出向量的模长（magnitude），即胶囊输出向量的欧几里德范数。
        
        这个操作将每个输出胶囊的向量表示变成一个标量，形状为 (batch_size, num_nodes, 1, 1)
        '''
        #(128, 10, 16, 1)

        # (batch,num_nodes,1,1?)
        zero = Variable(torch.zeros(1)).to(self.device)  # 为什么把0这么写
        m_plus = 0.9
        m_minus = 0.1
        '''
        分别表示边界的上界和下界。
        '''
        max_l = torch.max(m_plus - v_mag, zero).view(batch_size, -1) ** 2

        '''
        max_l：计算了边界 m_plus 和模长 v_mag 之差的最大值，用于计算边界损失。这个操作确保当模长小于 m_plus 时，边界损失为零。
        .view(batch_size, -1) 将张量的形状从 (batch_size, num_nodes, 1, 1) 改变为 (batch_size, num_nodes)
        '''
        max_r = torch.max(v_mag - m_minus, zero).view(batch_size, -1) ** 2
        '''
        max_r：计算了模长 v_mag 减去下界 m_minus 后的最大值，同样用于计算边界损失。这个操作确保当模长大于 m_minus 时，边界损失为零
        '''
        loss_lambda = 0.5
        T_c = target
        '''
        loss_lambda：边界损失的权重系数。
        T_c：表示目标标签 target
        '''
        # print(target.size()) 128,10
        L_c = T_c * max_l + loss_lambda * (1 - T_c) * max_r  # 为什么T_c-1不行，原来是0的地方变成了-1，我们需要1
        '''
        T_c * max_l 部分表示对于属于正确类别的输出胶囊，我们希望其长度（间隔）趋近于m_plus，而 (1 - T_c) * max_r 部分表示对于不属于正确类别的输出胶囊，我们希望其长度趋近于m_minus
        '''
        #计算边界损失 L_c，结合了上界和下界的计算，以及权重系数
        L_c = L_c.sum(dim=1)
        #对 L_c 在 num_nodes 维度上求和，得到每个样本的总边界损失

        if size_average:
            L_c = L_c.mean()
        return L_c
    '''
    计算重构损失在胶囊网络中，输入的胶囊输出向量的长度是一个重要特征。
    
    这里通过计算每个胶囊输出向量的长度，v_mag 存储了这些长度。
    
    对 input 中的每个元素平方后，通过 sum(dim=2) 求和，得到了每个胶囊输出向量的长度
    '''

    # def reconstruction_loss(self, images, input, size_average=True):
    #     pass
