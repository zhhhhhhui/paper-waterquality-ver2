import numpy as np

def create_image_cubes(X, window_size=28, stride=28):
    patches = []
    for r in range(0, X.shape[1] - window_size + 1, stride):
        for c in range(0, X.shape[2] - window_size + 1, stride):
            patch = X[:, r:r + window_size, c:c + window_size]
            if patch.shape == (X.shape[0], window_size, window_size):
                patches.append(patch)
    return np.array(patches)