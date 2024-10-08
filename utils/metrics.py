import torch
from sklearn.metrics import cohen_kappa_score, confusion_matrix


def accuracy(output, target):
    """
    计算批次的准确率
    Args:
        output: 模型的预测输出 (logits)
        target: 实际标签
    Returns:
        准确率 (acc)
    """
    pred = output.max(1, keepdim=True)[1]  # 获取预测类别
    correct = pred.eq(target.view_as(pred)).sum().item()  # 计算预测正确的数量
    acc = correct / target.size(0)  # 计算准确率
    return acc


def overall_accuracy(predictions, targets):
    """
    计算Overall Accuracy (OA), 总体准确率
    Args:
        predictions: 模型的预测类别
        targets: 实际类别
    Returns:
        Overall Accuracy
    """
    correct = (predictions == targets).sum()
    total = len(targets)
    return correct / total


def average_accuracy(predictions, targets, num_classes):
    """
    计算每个类别的准确率，并返回平均准确率 (AA)
    Args:
        predictions: 模型的预测类别
        targets: 实际类别
        num_classes: 类别数量
    Returns:
        平均准确率 (AA)
    """
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    for i in range(len(targets)):
        label = targets[i]
        pred = predictions[i]
        if label == pred:
            class_correct[label] += 1
        class_total[label] += 1

    # 计算每个类别的准确率
    class_accuracies = [class_correct[i] / class_total[i] if class_total[i] != 0 else 0 for i in range(num_classes)]

    # 返回所有类别的平均准确率
    return sum(class_accuracies) / num_classes


def kappa_score(predictions, targets, num_classes):
    """
    计算 Cohen's Kappa Score
    Args:
        predictions: 模型的预测类别
        targets: 实际类别
        num_classes: 类别数量
    Returns:
        Kappa系数
    """
    return cohen_kappa_score(targets, predictions, labels=[i for i in range(num_classes)])
