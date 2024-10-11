import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from scipy.ndimage import median_filter
import random
import torch


def normalize_data(X, method="minmax"):
    """
    Args:
        X
        method
    Returns:
    """
    if method == "minmax":
        scaler = MinMaxScaler()
    elif method == "standard":
        scaler = StandardScaler()
    else:
        raise ValueError(f"{method}")

    n_samples, n_rows, n_cols = X.shape
    X_reshaped = X.reshape(n_samples, n_rows * n_cols)
    X_normalized = scaler.fit_transform(X_reshaped)
    return X_normalized.reshape(n_samples, n_rows, n_cols)


def reflectance_calibration(X, solar_irradiance, solar_angle):
    """
    Args:
        X(n_bands, height, width)
        solar_irradiance
        solar_angle
    Returns:
    """
    reflectance = (X * np.pi) / (solar_irradiance * np.cos(solar_angle))
    reflectance = np.clip(reflectance, 0, 1)
    return reflectance


def atmospheric_correction(X, method='QUAC'):
    """
    Args:
        X:  (n_bands, height, width)
        method:
    Returns:
    """
    if method == 'QUAC':

        X_corrected = X - np.mean(X, axis=0)
    elif method == 'FLAASH':

        pass
    else:
        raise ValueError(f"{method}")

    return X_corrected


def geometric_correction(X, geo_transform):
    """

    Args:
        X (n_bands, height, width)
        geo_transform:
    Returns:
    """
    corrected_image = cv2.warpAffine(X, geo_transform, (X.shape[2], X.shape[1]))
    return corrected_image


def denoise_image(X, method="median", kernel_size=3):
    """

    Args:
        X (n_bands, height, width)
        method: "median"、"mean" OR "gaussian"
        kernel_size:
    Returns:
    """
    if method == "median":
        denoised = np.array([median_filter(X[b], size=kernel_size) for b in range(X.shape[0])])
    elif method == "mean":
        denoised = np.array([cv2.blur(X[b], (kernel_size, kernel_size)) for b in range(X.shape[0])])
    elif method == "gaussian":
        denoised = np.array([cv2.GaussianBlur(X[b], (kernel_size, kernel_size), 0) for b in range(X.shape[0])])
    else:
        raise ValueError(f"{method}")

    return denoised


def preprocess_image(X, pca_components=None, window_size=28, stride=28,
                     normalization="minmax", augment=False, selected_bands=None,
                     solar_irradiance=None, solar_angle=None, geo_transform=None,
                     atmospheric_method='QUAC', denoise_method='median'):
    X = normalize_data(X, method=normalization)

    if solar_irradiance and solar_angle:
        X = reflectance_calibration(X, solar_irradiance, solar_angle)

    X = atmospheric_correction(X, method=atmospheric_method)

    if geo_transform:
        X = geometric_correction(X, geo_transform)

    X = denoise_image(X, method=denoise_method)

    if pca_components:
        X = apply_pca(X, pca_components)

    if augment:
        X = random_flip(X)
        X = random_rotation(X)

    return patches[:, np.newaxis, :, :, :]
