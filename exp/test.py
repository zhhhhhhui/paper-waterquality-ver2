import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


def test(network, test_loader, to_one_hot):
    """
    Args:
        network
        test_loader
        to_one_hot
    """
    network.eval()  # dropout
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            target_indices = target
            target_one_hot = to_one_hot(target_indices, network.digits.num_node)

            # GPU
            data, target = Variable(data).cuda(), Variable(target_one_hot).cuda()
            output = network(data)

            # reduction='sum' replace size_average=False
            test_loss += network.loss(data, output, target, reduction='sum').item()

            v_mag = torch.sqrt((output ** 2).sum(dim=2, keepdim=True))
            pred = v_mag.data.max(1, keepdim=True)[1].cpu()

            correct += pred.eq(target_indices.view_as(pred)).sum().item()
            total += target_indices.size(0)  # update

    # average loss
    test_loss /= total

    # accuracy
    accuracy = 100. * correct / total

    # print
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)\n')

    return test_loss, accuracy


