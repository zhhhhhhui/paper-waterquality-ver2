import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


def test(network, test_loader, to_one_hot):
    """
    测试模型在测试集上的表现
    Args:
        network: 训练好的模型
        test_loader: 测试集的数据加载器
        to_one_hot: 将标签转换为 one-hot 编码的函数
    """
    network.eval()  # 设置模型为评估模式，关闭dropout等操作
    test_loss = 0
    correct = 0
    total = 0  # 记录总样本数

    with torch.no_grad():  # 在推理模式下不需要计算梯度
        for data, target in test_loader:
            target_indices = target
            target_one_hot = to_one_hot(target_indices, network.digits.num_node)

            # 将数据转移到GPU
            data, target = Variable(data).cuda(), Variable(target_one_hot).cuda()

            # 前向传播，获取模型输出
            output = network(data)

            # 计算损失，使用 reduction='sum' 来替代 size_average=False
            test_loss += network.loss(data, output, target, reduction='sum').item()

            # 计算胶囊网络的长度并获取预测结果
            v_mag = torch.sqrt((output ** 2).sum(dim=2, keepdim=True))
            pred = v_mag.data.max(1, keepdim=True)[1].cpu()  # 获取预测类别

            # 统计预测正确的数量
            correct += pred.eq(target_indices.view_as(pred)).sum().item()
            total += target_indices.size(0)  # 更新总样本数

    # 计算平均损失
    test_loss /= total

    # 计算总体准确率
    accuracy = 100. * correct / total

    # 打印测试结果
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)\n')

    return test_loss, accuracy

# 示例：假设您已经定义了模型 `network`，测试数据加载器 `test_loader` 和 `to_one_hot` 函数
# test(network, test_loader, to_one_hot)
