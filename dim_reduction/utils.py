import numpy as np

def pca_compression_ratio(original_shape, n_components):
    """
    Compute PCA compression ratio for an image reshaped into (n_samples, n_features).

    Args:
        original_shape (tuple): Shape of the centered data (n_samples, n_features).
        n_components (int): Number of principal components kept.

    Returns:
        dict: {
            "original_size": number of values in original data,
            "compressed_size": number of values after PCA (proj_data + eigenvectors),
            "ratio": compressed_size / original_size
        }
    """
    n_samples, n_features = original_shape

    # original storage = full matrix
    original_size = n_samples * n_features

    # PCA storage = projected data + eigenvectors
    compressed_size = (n_samples * n_components) + (n_features * n_components)

    ratio = compressed_size / original_size

    return {
        "original_size": original_size,
        "compressed_size": compressed_size,
        "ratio": ratio
    }
