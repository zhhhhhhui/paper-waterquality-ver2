import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()


# 读取Excel文件
data = pd.read_excel('C:/Users/zh/Desktop/export/accuracy/CSA-net.xlsx')

# 提取轮次和精度值列
rounds = data.iloc[:, 0]
cnn_accuracy = data.iloc[:, 1]
hybridnet_accuracy = data.iloc[:, 2]

# 创建绘图对象
fig, ax = plt.subplots()

# 绘制*各个对比模型的精度折线
ax.plot(rounds, cnn_accuracy, marker='o', linestyle='-', color='blue', label='CSANet')

# 绘制HybridNet精度折线
ax.plot(rounds, hybridnet_accuracy, marker='o', linestyle='-', color='red', label='HybridNet')

# 添加散点图
ax.scatter(rounds, cnn_accuracy, color='red')
ax.scatter(rounds, hybridnet_accuracy, color='blue')

# 设置坐标轴标签
ax.set_xlabel('Epoch',fontsize=14)
ax.set_ylabel('Accuracy',fontsize=14)

# 设置坐标轴刻度的字体大小
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)

# 设置图例
ax.legend()

# 显示图表
plt.show()