import os
import torch
import rasterio
import numpy as np
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from hybrid_network import HybridNet


def load_image(image_file, pca_components=30, window_size=28, stride=28):
    """

    Args:
        image_file
        pca_components
        window_size
        stride
    Returns:
        3D HSI
    """

    with rasterio.open(image_file) as src:
        image_data = src.read()
        n_samples, n_rows, n_columns = image_data.shape

        image_data_reshaped = image_data.reshape(n_samples, n_rows * n_columns)
        scaler = StandardScaler()
        X_standardized = scaler.fit_transform(image_data_reshaped)

        X_standardized = X_standardized.reshape(n_samples, n_rows, n_columns)

        X_standardized_pca = applyPCA(X_standardized, numComponents=pca_components)

        patches = createImageCubes(X_standardized_pca, windowSize=window_size, stride=stride)

    patches = patches[:, np.newaxis, :, :, :]
    return patches


def validate(model, data_loader, device):
    """
    Args:
        model
        data_loader
        device（CPU OR GPU）
    """
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            data = data.to(device)
            output = model(data)

            v_mag = torch.sqrt((output ** 2).sum(dim=2, keepdim=True))
            pred = v_mag.data.max(1, keepdim=True)[1].cpu()

            for j in range(pred.size(0)):
                predicted_class = pred[j, 0, 0, 0].item()
                print(f" {i * pred.size(0) + j + 1} : {predicted_class}")


def main():
    image_file = ''

    pca_components = 30
    window_size = 28
    stride = 28

    image_patches = load_image(image_file, pca_components, window_size, stride)

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

    model.load_state_dict(torch.load('model.pth'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    data_loader = DataLoader(torch.tensor(image_patches, dtype=torch.float32), batch_size=32)

    validate(model, data_loader, device)


if __name__ == '__main__':
    main()
