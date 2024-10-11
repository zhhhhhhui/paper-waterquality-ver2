import torch
from sklearn.metrics import cohen_kappa_score, confusion_matrix


def accuracy(output, target):
    """
    Args:
        output: logits
        target
    Returns:
        acc
    """
    pred = output.max(1, keepdim=True)[1]
    correct = pred.eq(target.view_as(pred)).sum().item()
    acc = correct / target.size(0)
    return acc


def overall_accuracy(predictions, targets):
    """
    Overall Accuracy (OA)
    Args:
        predictions
        targets
    Returns:
        Overall Accuracy
    """
    correct = (predictions == targets).sum()
    total = len(targets)
    return correct / total


def average_accuracy(predictions, targets, num_classes):
    """

    Args:
        predictions
        targets
        num_classes
    Returns:
        AA
    """
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    for i in range(len(targets)):
        label = targets[i]
        pred = predictions[i]
        if label == pred:
            class_correct[label] += 1
        class_total[label] += 1

    class_accuracies = [class_correct[i] / class_total[i] if class_total[i] != 0 else 0 for i in range(num_classes)]

    return sum(class_accuracies) / num_classes


def kappa_score(predictions, targets, num_classes):
    """
    Cohen's Kappa Score
    Args:
        predictions
        targets
        num_classes
    Returns:
        Kappa
    """
    return cohen_kappa_score(targets, predictions, labels=[i for i in range(num_classes)])
