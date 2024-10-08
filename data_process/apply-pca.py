import numpy as np
from sklearn.decomposition import PCA

def apply_pca(X, num_components=30):
    reshaped_X = np.reshape(X, (-1, X.shape[0]))
    pca = PCA(n_components=num_components, whiten=True)
    transformed_X = pca.fit_transform(reshaped_X)
    return np.reshape(transformed_X, (num_components, X.shape[1], X.shape[2]))