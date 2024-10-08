import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from scipy.ndimage import median_filter
import random
import torch


def normalize_data(X, method="minmax"):
    """
    对输入的高光谱数据进行归一化
    Args:
        X: 输入的高光谱数据
        method: 归一化方法，"minmax" 或 "standard"
    Returns:
        归一化后的数据
    """
    if method == "minmax":
        scaler = MinMaxScaler()
    elif method == "standard":
        scaler = StandardScaler()
    else:
        raise ValueError(f"未知归一化方法：{method}")

    n_samples, n_rows, n_cols = X.shape
    X_reshaped = X.reshape(n_samples, n_rows * n_cols)
    X_normalized = scaler.fit_transform(X_reshaped)
    return X_normalized.reshape(n_samples, n_rows, n_cols)


def reflectance_calibration(X, solar_irradiance, solar_angle):
    """
    对高光谱图像进行反射率校准，将DN值转化为物理量反射率
    Args:
        X: 输入的高光谱数据 (n_bands, height, width)
        solar_irradiance: 太阳辐照度
        solar_angle: 太阳角度（弧度）
    Returns:
        校准后的反射率数据
    """
    # 假设每个波段都有对应的太阳辐照度常数
    reflectance = (X * np.pi) / (solar_irradiance * np.cos(solar_angle))
    reflectance = np.clip(reflectance, 0, 1)  # 限制反射率值在0到1之间
    return reflectance


def atmospheric_correction(X, method='QUAC'):
    """
    对高光谱图像进行大气校正
    Args:
        X: 输入的高光谱数据 (n_bands, height, width)
        method: 大气校正方法，默认为QUAC
    Returns:
        大气校正后的数据
    """
    if method == 'QUAC':
        # 简单的大气校正模拟
        X_corrected = X - np.mean(X, axis=0)  # 减去平均值以模拟大气校正
    elif method == 'FLAASH':
        # 假设实现了FLAASH大气校正方法
        pass
    else:
        raise ValueError(f"未知的大气校正方法：{method}")

    return X_corrected


def geometric_correction(X, geo_transform):
    """
    对图像进行几何校正
    Args:
        X: 输入的高光谱数据 (n_bands, height, width)
        geo_transform: 几何校正的变换矩阵
    Returns:
        几何校正后的数据
    """
    # 假设已实现几何校正的函数
    # 这里可以使用OpenCV或者GDAL等库进行几何变换
    corrected_image = cv2.warpAffine(X, geo_transform, (X.shape[2], X.shape[1]))
    return corrected_image


def denoise_image(X, method="median", kernel_size=3):
    """
    对图像进行去噪
    Args:
        X: 输入的高光谱数据 (n_bands, height, width)
        method: 去噪方法，"median"、"mean" 或 "gaussian"
        kernel_size: 滤波器的大小
    Returns:
        去噪后的图像
    """
    if method == "median":
        denoised = np.array([median_filter(X[b], size=kernel_size) for b in range(X.shape[0])])
    elif method == "mean":
        denoised = np.array([cv2.blur(X[b], (kernel_size, kernel_size)) for b in range(X.shape[0])])
    elif method == "gaussian":
        denoised = np.array([cv2.GaussianBlur(X[b], (kernel_size, kernel_size), 0) for b in range(X.shape[0])])
    else:
        raise ValueError(f"未知的去噪方法：{method}")

    return denoised


def preprocess_image(X, pca_components=None, window_size=28, stride=28,
                     normalization="minmax", augment=False, selected_bands=None,
                     solar_irradiance=None, solar_angle=None, geo_transform=None,
                     atmospheric_method='QUAC', denoise_method='median'):
    """
    对高光谱图像进行预处理，包括归一化、反射率校准、大气校正、去噪和数据增强

    """
    # 归一化
    X = normalize_data(X, method=normalization)



    # 反射率校准
    if solar_irradiance and solar_angle:
        X = reflectance_calibration(X, solar_irradiance, solar_angle)

    # 大气校正
    X = atmospheric_correction(X, method=atmospheric_method)

    # 几何校正
    if geo_transform:
        X = geometric_correction(X, geo_transform)

    # 去噪
    X = denoise_image(X, method=denoise_method)

    # PCA降维
    if pca_components:
        X = apply_pca(X, pca_components)

    # 数据增强
    if augment:
        X = random_flip(X)
        X = random_rotation(X)


    # 增加额外维度以适配3D卷积
    return patches[:, np.newaxis, :, :, :]
