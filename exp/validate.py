import os
import torch
import rasterio
import numpy as np
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from hybrid_network import HybridNet  # 导入你定义的模型




def load_image(image_file, pca_components=30, window_size=28, stride=28):
    """
    加载并处理图像，包括标准化、PCA降维和滑动窗口分割
    Args:
        image_file: 图像文件路径
        pca_components: 主成分分析的维度
        window_size: 滑动窗口大小
        stride: 滑动窗口步长
    Returns:
        处理后的3D高光谱图像数据
    """
    # 打开图像
    with rasterio.open(image_file) as src:
        image_data = src.read()  # 加载图像数据
        n_samples, n_rows, n_columns = image_data.shape

        # 将数据标准化
        image_data_reshaped = image_data.reshape(n_samples, n_rows * n_columns)
        scaler = StandardScaler()
        X_standardized = scaler.fit_transform(image_data_reshaped)

        # 还原为原始形状
        X_standardized = X_standardized.reshape(n_samples, n_rows, n_columns)

        # 应用PCA降维
        X_standardized_pca = applyPCA(X_standardized, numComponents=pca_components)

        # 使用滑动窗口分割图像为patch
        patches = createImageCubes(X_standardized_pca, windowSize=window_size, stride=stride)

    # 为了兼容3D CNN，增加一个额外的维度
    patches = patches[:, np.newaxis, :, :, :]
    return patches


def validate(model, data_loader, device):
    """
    验证模型的表现
    Args:
        model: 训练好的模型
        data_loader: 验证集的数据加载器
        device: 使用的设备（CPU或GPU）
    """
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            data = data.to(device)
            output = model(data)

            # 计算输出张量的模长
            v_mag = torch.sqrt((output ** 2).sum(dim=2, keepdim=True))
            pred = v_mag.data.max(1, keepdim=True)[1].cpu()

            # 打印每个样本的预测类别
            for j in range(pred.size(0)):
                predicted_class = pred[j, 0, 0, 0].item()
                print(f"样本 {i * pred.size(0) + j + 1} 的预测类别: {predicted_class}")


def main():
    # 定义图像文件路径
    image_file = 'D:/pythonproject/work/高光谱/鱼塘8.tif'

    # 定义参数
    pca_components = 30
    window_size = 28
    stride = 28

    # 加载和处理图像
    image_patches = load_image(image_file, pca_components, window_size, stride)

    # 加载预训练的模型
    model = HybridNet(
        image_width=28,
        image_height=28,
        image_channels=30,
        conv_input_channel=30,
        conv_output_channel=576,
        num_primary_node=32 * 6 * 6,
        primary_node_size=8,
        num_output_node=3,
        output_node_size=16
    )

    # 加载模型参数
    model.load_state_dict(torch.load('model.pth'))

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 将处理好的图像patch转换为数据加载器
    data_loader = DataLoader(torch.tensor(image_patches, dtype=torch.float32), batch_size=32)

    # 验证模型
    validate(model, data_loader, device)


if __name__ == '__main__':
    main()
