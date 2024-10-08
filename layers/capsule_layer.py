import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn.functional as F



class ConvUnit(nn.Module):
    def __init__(self, in_channels):  # 在创建 ConvUnit 类的实例时调用。它接受一个参数

        super(ConvUnit, self).__init__()  # 调用了父类 nn.Module 的构造函数，确保正确地初始化 ConvUnit 类的实例。
        # 通过调用父类的构造函数，可以在子类中使用 nn.Module 提供的各种功能和属性

        # self.conv0 = nn.Conv2d(in_channels=in_channels, #256
        #                        out_channels=32,  # 后面no_routing的时候用了8个单元拼接
        #                        kernel_size=9,
        #                        stride=2,
        #                        bias=True)
        # [24,832,50,50]
        self.conv0 = nn.Conv2d(in_channels=in_channels,
                               out_channels=32,
                               kernel_size=9,
                               stride=2,
                               bias=True)
        # [24,32,22,22]
    def forward(self, x):
        return self.conv0(x)


'''
实现了一个包含动态路由和非动态路由机制的胶囊层。它根据不同的模式执行胶囊之间的信息传递和关联性建模
'''


class CapsuleLayer(nn.Module):
    def __init__(self, in_node_num, in_size, num_node, node_size, use_routing):
        '''
        in_node_num: 输入胶囊的数量。它表示在当前胶囊层中，上一层的输出被认为是多少个输入胶囊。

        in_size: 输入胶囊的维度（大小）。它表示每个输入胶囊的向量维度或特征数量。

        num_node: 输出胶囊的数量。它表示当前层的胶囊将生成多少个输出胶囊。

        node_size: 输出胶囊的维度（大小）。它表示每个输出胶囊的向量维度或特征数量。

        use_routing: 是否使用动态路由机制。如果为 True，则意味着胶囊之间将通过动态路
        '''
        super(CapsuleLayer, self).__init__()

        self.in_node_num = in_node_num #输入胶囊的数量
        self.in_size = in_size #输入胶囊的维度（大小）
        self.num_node = num_node #输出胶囊的数量
        self.use_routing = use_routing
        self.node_size = node_size #输出胶囊的维度（大小）
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.use_routing:
            #self情况(32 * 6 * 6, 8, 10, 16)
            self.W = nn.Parameter(torch.randn(1, in_node_num, num_node, node_size, in_size))  # *0.03

            # W.shape(1,32*6*6,10,16,8) why 1 here
            #这里顺序调换了，(输入胶囊数量，输出胶囊数量，输出维度，输入维度)
        else:
            # 没有实例化时的x.shape的形状是 (128, 256, 20, 20) 来自上一层的输出
            # self情况(0,256,32*6*6,8)
            '''
            如果 use_routing 为 False，则会创建 node_size 个 ConvUnit 实例，其中 ConvUnit 是一个自定义的卷积单元
            
            这个函数用于创建一个 ConvUnit 实例，并将其作为 CapsuleLayer 的子模块添加。
            
            在 without routing 的情况下，每个输出胶囊将使用一个 ConvUnit 来执行卷积操作
            '''
            def create_conv_unit(unit_idx):
                unit = ConvUnit(in_channels=in_size)
                self.add_module('unit_' + str(unit_idx), unit)  # 创造出ConvUnit的子模块
                return unit

            self.units = [create_conv_unit(i) for i in range(node_size)]
            # 一共8个卷积单元，在without routing的情况下
            '''
            输出特征图的形状为，(128, 32, 6, 6)
            这里的32是输出通道数，6x6是输出特征图的空间分辨率
            '''

    @staticmethod
    def squash(s):
        #变量 s 是一个表示胶囊网络中胶囊向量的张量（tensor)
        # s.shape ? (batch,node_size,num) num是原来每个输出胶囊的特征图展平为一个向量32*6*6
        # print(s.type())
        mag_sq = torch.sum(s ** 2, dim=1, keepdim=True)
        # print(s.size())
        #得到(batch, num)
        mag = torch.sqrt(mag_sq)
        # 计算每个胶囊向量的模值
        s = (mag_sq / (1.0 + mag_sq)) * (s / mag)
        '''
        首先，它将模值平方 mag_sq 进行归一化，使其在 [0, 1] 范围内。
        
        然后，将胶囊向量 s 进行归一化，将其除以模值 mag。
        
        最后，将归一化的模值与归一化的胶囊向量相乘，得到压缩后的胶囊向量。
        
        这个操作确保输出胶囊的模值范围在 [0, 1] 内，同时保持了方向信息
        
        (s / mag) 的计算将输出胶囊的每个维度都除以其对应的模值，实现了对输出胶囊向量的归一化。
        
        这样做的目的是为了确保胶囊网络的输出向量在模值上保持一致，而不同的模值会导致信息的扭曲。
        
        通过将模值标准化为1，网络可以更好地表达特征的存在与否以及方向
        '''
        return s
        # 函数返回压缩后的胶囊向量 s，它的形状与输入相同，为 (batch, node_size, num)

    def forward(self, x):
        if self.use_routing:
            return self.routing(x)
        else:
            return self.no_routing(x)


    def no_routing(self, x):#
        # self情况(0,256,32*6*6,8)
        #输出特征图的形状为(128, 32, 6, 6) 总共创建了8个
        #输出都是一个形状为 (batch_size, in_size, height, width) 的特征图
        # 分离卷积单元和squash
        # x(batch,channel256,height,weight)
        """
                x(batch_size, channels, height, width)，
                其中：batch_size：表示批次中图像的数量。
                channels：表示图像的通道数，通常为 3（彩色图像）或 1（灰度图像）。
                height：表示图像的高度，即图像的垂直像素数。
                width：表示图像的宽度，即图像的水平像素数。
                所以，x(batch, channel, height, width) 表示一个批次中的一张图像，其中包含了 channel 个通道，每个通道的尺寸为 height（高度）和 width（宽度）。在胶囊网络中，这样的图像数据将通过不同的卷积和胶囊层进行处理和特征提取
        """
        u = [self.units[i](x) for i in range(self.node_size)]

        u = torch.stack(u, dim=1)
        '''
        #将这 8 个输出胶囊的特征图堆叠在一起，通过 torch.stack(u, dim=1)，形成一个新的特征图 u，
        # 其形状为 (batch_size, node_size, in_size, height, width)
        (128,32,6,6)
        '''
        # u(batch,in_size8,channel32,height,weight)
        #(128,8,32,6,6)
        # flatten(batch,unit_size8,-1)
        u = u.view(x.size(0), self.node_size, -1)
        #u(128,8,32*6*6)
        '''
        这个操作类似于将二维的图像特征转化为一维的特征向量，使得胶囊网络可以更好地处理和学习这些特征，以及将它们传递给后续的层进行分类、重建等任务。
        
        展平操作是为了将每个胶囊产生的特征表示转换为更紧凑的形式，方便后续的处理和分析
        '''
        return CapsuleLayer.squash(u)
        #对输入的胶囊向量进行非线性压缩,形状不会变
        '''
                对 u 进行 reshape 操作，将其形状变为 (batch_size, node_size, -1)。这相当于将每个输出胶囊的特征图展平为一个向量。

                对展平后的 u 进行 squash 操作，使用 CapsuleLayer.squash(u)。

                这个操作将确保输出胶囊的向量长度保持在 0 到 1 之间，将向量进行归一化


                '''

    def routing(self, x):
        #x(batch_size, node_size, in_size * height * width)
        # (128,8,32*6*6)
        batch_size = x.size(0)
        x = x.transpose(1, 2)  # numpy和torch的transpose不同，torch纯粹是两个维度的交换
        '''
        我理解的交换的意思是原来的in_size * height * width是上一层的输出，现在要作为这一层的输入，所以要交换顺序
        '''
        #x = x.transpose(1, 2): 将输入张量 x 的维度 1 和维度 2 进行交换。这是为了将输入的节点数和输出的节点数对应起来
        x = torch.stack([x] * self.num_node, dim=2).unsqueeze(4)  # 将array复制num份放入列表，再stack

        # (batch,num_nodes,out_num_nodes,node_size8,1)
        #(batch_size, in_size * height * width, self.num_node, node_size, 1)
        #(128,32*6*6,10,8,1)
        #self.num_node输出胶囊的数量为10，in_size * height * width作为这一层的输入
        '''
        torch.stack([x] * self.num_node, dim=2)：这将张量 x 复制 self.num_node 次，并在维度 2 上堆叠这些副本。

        这将创建一个新的张量，形状为 (batch_size, in_size * height * width, self.num_node, node_size)。

        在这个操作中，原本的节点数在新的维度上被复制了 self.num_node 次

        unsqueeze(4): 这个操作是在维度 4 上插入一个新维度，即在堆叠后的张量中的最后一个维度。

        这将创建一个新的维度，其大小为 1。这么做是为了与后续的矩阵乘法进行兼容
        '''
        # (batch,num_nodes,out_num_nodes,node_size8,1)

        W = torch.cat([self.W] * batch_size, dim=0) * 0.03  # 0.03很重要保证数值稳定
        '''
        # W.shape(1,32*6*6,10,16,8) why 1 here
        X(128,32*6*6,10,8,1),W(128,32*6*6,10,16,8)
        '''
        # (batch,num_nodes,out_num_nodes,out_node_size16,node_size8)

        u_hat = torch.matmul(W, x)  # 智能矩阵乘法
        #预测胶囊的输出向量
        '''
        而矩阵乘法要求输入张量的最后一个维度和另一个矩阵的倒数第二个维度匹配,W最后一个为8，所以X的倒数第二个也是8，可以矩阵乘法
        '''
        # test0 = torch.sum(u_hat ** 2, dim=3)
        #计算出每个输出胶囊的预测值 u_hat
        # (batch,num_nodes,out_num_nodes,out_node_size16,1)

        b_ij = torch.zeros(1, self.in_node_num, self.num_node, 1).to(self.device)
        '''
        初始化胶囊之间的关联性权重 b_ij，其形状为 (1, in_node_num, num_node, 1)
        (1,32*6*6,10,1)，其中分别为1，输入胶囊，输出胶囊，1
        
        Variable 和 .cuda() 是用于将 b_ij 移动到 GPU 上的步骤
        '''
        num_iterations = 3
        for iteration in range(num_iterations):
            # print(b_ij.size())
            c_ij = F.softmax(b_ij, dim=2)  # 默认dim=i,！感觉有错误，应该对j...但这样会使胶囊永远接近1
            '''
            softmax是指数运算
            归一化。这样做是为了将关联性权重转化为概率值，表示输出胶囊与输入胶囊的匹配程度
            '''
            # if iteration==0:
            # c_ij[:,:,:,:]=0.0009
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)

            # 故技重施,(batch,num_nodes,out_num_nodes,1,1)
            #c_ij(128,32*6*6,10,1,1)

            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            #(c_ij * u_hat) 并不是矩阵乘法，而是逐元素相乘
            #这里应该是用了广播
            # (batch,1,out_num_nodes,out_node_size,1)为什么要有最后一维为1
            '''
            c_ij(128,32*6*6,10,1,1)
            u_hat(128,32*6*6,10,16,1)
            利用归一化权重 c_ij 对预测值 u_hat 进行加权求和，得到输出胶囊的加权输入
            (batch,1,10,16,1)
            s_j 表示每个输出胶囊的加权输入，其中关联性权重 c_ij 被用来将输出胶囊的预测值乘以输入胶囊之间的关联性，
            然后在输入胶囊维度上求和得到加权输入。这个加权输入将在后续步骤中用于计算输出胶囊的输出向量
            '''
            #s_j(128,1,10,16,1)
            v_j = CapsuleLayer.squash(s_j.squeeze().transpose(1, 2))
            '''
            首先，从之前计算得到的加权输入 s_j 中移除尺寸为 1 的维度，即通过 squeeze() 去除
            通过 transpose(1, 2) 交换第 1 和第 2 个维度，即将输出胶囊的数量移到前面。
            最后，通过 CapsuleLayer.squash() 函数对加权输入进行非线性压缩，得到输出胶囊的输出向量 v_j
            '''
            '''
             使用关联性权重对输入胶囊的预测(u_hat)进行加权求和，得到输出胶囊的加权输入。
             
             这样，每个输出胶囊都可以聚焦于与其相关的输入特征，从而更好地捕捉对象的属性和关系
            '''
            # test=torch.sum(v_j**2,dim=1)
            v_j = v_j.transpose(1, 2)[:, None, :, :, None]
            #操作会在维度 1 和 4 上插入新的维度
            v_j1 = torch.cat([v_j] * self.in_node_num, dim=1)

            # print(v_j1.size())
            # (batch,num_nodes,out_num_nodes,out_node_size,1)
            #(128,32*6*6,10,16,1)
            #u_hat(128,32*6*6,10,16,1)->(128,32*6*6,10,1,16)
            u_vj1 = torch.matmul(u_hat.transpose(3, 4), v_j1).squeeze(4).mean(dim=0, keepdim=True)
            '''
            通过 squeeze(4) 去除最后一个维度，得到加权输出。
            最后，使用 mean(dim=0, keepdim=True) 对所有输入样本进行平均，得到一个用于更新关联性权重 b_ij 的增量 b_ij = b_ij + u_vj1
            u_vj1(1,32*6*6,10,1)
            '''
            b_ij = b_ij + u_vj1
            '''
            为什么用加法：
            
            迭代更新： 动态路由算法是一个迭代过程，胶囊之间的关联性需要在多次迭代中逐渐调整。使用累积的方式（加法）可以在每次迭代中保留之前的更新，以便更好地进行调整。

            避免剧烈变化： 如果只使用当前一次迭代的 u_vj1，会导致关联性在每次迭代中剧烈变化，可能导致不稳定的训练和收敛过程。通过累积更新，可以在多次迭代中平滑地调整关联性，有助于稳定训练。

            集体决策： 胶囊网络的目标之一是实现胶囊之间的集体决策，而不仅仅是依赖于单个迭代步骤的更新。累积更新可以帮助胶囊之间共同协作，以便在多次迭代中决定输出胶囊与输入胶囊的关联性
            '''
        # test = torch.sum(v_j.squeeze() ** 2, dim=2)
        return v_j.squeeze(1)
        '''
        这个最终的张量代表了胶囊网络对于每个样本在每个输出胶囊上的预测输出。
        
        这个张量中的每个条目可以被解释为每个输入样本的每个输出胶囊的“激活”或“强度”。
        
        通常用于在分类任务中进行预测或进一步的处理
        (128, 10, 16, 1)
        '''
