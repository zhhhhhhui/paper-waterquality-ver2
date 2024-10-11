import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()


data = pd.read_excel('C:/Users/zh/Desktop/export/accuracy/CSA-net.xlsx')

rounds = data.iloc[:, 0]
cnn_accuracy = data.iloc[:, 1]
hybridnet_accuracy = data.iloc[:, 2]

fig, ax = plt.subplots()

ax.plot(rounds, cnn_accuracy, marker='o', linestyle='-', color='blue', label='CSANet')

ax.plot(rounds, hybridnet_accuracy, marker='o', linestyle='-', color='red', label='HybridNet')

ax.scatter(rounds, cnn_accuracy, color='red')
ax.scatter(rounds, hybridnet_accuracy, color='blue')

ax.set_xlabel('Epoch', fontsize=14)
ax.set_ylabel('Accuracy', fontsize=14)

ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)

ax.legend()
plt.show()
