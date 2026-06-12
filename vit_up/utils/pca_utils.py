"""PCA utilities for keypoint features"""

import torch
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA
import numpy as np

from . import image_processing


def pca_iterative(X, n_components=None, batch_size=1000, std_normalize=True):
    """
    Perform iterative PCA on the input data using sklearn's IncrementalPCA.

    Args:
        X (np.ndarray): Input array of shape (n, d), where n is the number of samples and d is the feature dimension.
        n_components (int, optional): Number of principal components to keep. If None, all components are kept.
        batch_size (int): Size of batches for iterative fitting.
        std_normalize (bool): Whether to standardize the data.

    Returns:
        np.ndarray: PCA components of shape (n, n_components) if n_components is provided, otherwise (n, d).
        np.ndarray: The original mean of the data.
        np.ndarray: The original standard deviation of the data (None if std_normalize=False).
        np.ndarray: The eigenvalues (explained variance).
        np.ndarray: The eigenvectors (principal components).
    """
    # Step 1: Compute statistics for normalization
    original_mean = np.mean(X, axis=0)
    original_std = None

    if std_normalize:
        original_std = np.std(X, axis=0)
        # Normalize the data
        X_normalized = (X - original_mean) / original_std
    else:
        # Only center the data
        X_normalized = X - original_mean

    # Step 2: Initialize IncrementalPCA
    if n_components is None:
        n_components = min(X.shape[0], X.shape[1])

    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)

    # Step 3: Fit PCA iteratively
    n_samples = X_normalized.shape[0]
    for i in tqdm(range(0, n_samples, batch_size)):
        batch_end = min(i + batch_size, n_samples)
        batch = X_normalized[i:batch_end]
        ipca.partial_fit(batch)

    # Step 4: Transform the data
    X_pca = ipca.transform(X_normalized)

    # Step 5: Extract components and eigenvalues
    eigenvalues = ipca.explained_variance_
    pca_eigenvectors = ipca.components_.T  # Shape: (d, n_components)

    return X_pca, original_mean, original_std, eigenvalues, pca_eigenvectors


def get_feature_mask(seg_mask: torch.Tensor, embd_size: int, threshold=0.2):
    h, w = seg_mask.shape
    pad_h, pad_w = image_processing.get_pad_sizes_from_img_shape(h, w)
    patch_size = max(h, w) / embd_size
    ft_mask = torch.zeros((embd_size, embd_size), dtype=torch.bool)
    for x in range(embd_size):
        for y in range(embd_size):
            x_orig, y_orig = (
                int(x * patch_size) - pad_w // 2,
                int(y * patch_size) - pad_h // 2,
            )
            x_orig_min, y_orig_min = max(x_orig, 0), max(y_orig, 0)
            x_orig_max, y_orig_max = int(x_orig + patch_size), int(y_orig + patch_size)
            x_orig_max, y_orig_max = max(x_orig_max, 0), max(y_orig_max, 0)
            if (
                torch.sum(seg_mask[y_orig_min:y_orig_max, x_orig_min:x_orig_max])
                / (patch_size**2)
                > threshold
            ):
                ft_mask[y, x] = True
    return ft_mask


def pca(X, k=None, std_normalize=True, weights=None):
    """
    Perform (optionally weighted) PCA on the input data.

    Args:
        X (torch.Tensor): (n, d) data matrix.
        k (int, optional): Number of components to keep. If None -> keep all.
        std_normalize (bool): Whether to divide by per-dim std.
        weights (torch.Tensor, optional): (n,) weights. If None -> uniform weights.

    Returns:
        X_pca: (n, k or d) projected data
        X_mean: (d,) weighted mean
        X_std:  (d,) weighted std (if std_normalize)
        eigenvalues: (d,)
        eigenvectors: (d, d) or (d, k)
    """

    n, d = X.shape

    # ------------------------------------------------------------------
    # 1) Prepare weights
    # ------------------------------------------------------------------
    if weights is None:
        w = torch.ones(n, device=X.device, dtype=X.dtype)
    else:
        w = weights.to(X.device, dtype=X.dtype)
        assert w.ndim == 1 and w.shape[0] == n, "weights must be shape (n,)"

    w = w / w.sum()  # normalize (not required but stable)

    # ------------------------------------------------------------------
    # 2) Weighted mean
    # ------------------------------------------------------------------
    X_mean = (w[:, None] * X).sum(dim=0)

    # ------------------------------------------------------------------
    # 3) Center data
    # ------------------------------------------------------------------
    X_hat = X - X_mean

    # ------------------------------------------------------------------
    # 4) Weighted std-normalization (optional)
    # ------------------------------------------------------------------
    if std_normalize:
        # weighted variance: Var = Σ w_i * (x_i - μ)^2
        var = (w[:, None] * X_hat * X_hat).sum(dim=0)
        X_std = torch.sqrt(var + 1e-8)
        X_hat = X_hat / X_std
    else:
        X_std = torch.ones(d, device=X.device, dtype=X.dtype)

    # ------------------------------------------------------------------
    # 5) Weighted covariance matrix
    #     C = Σ w_i x_i x_i^T
    # ------------------------------------------------------------------
    covariance_matrix = (X_hat.T * w) @ X_hat

    # ------------------------------------------------------------------
    # 6) Eigen decomposition
    # ------------------------------------------------------------------
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)

    # Sort descending
    sorted_idx = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[sorted_idx]
    eigenvectors = eigenvectors[:, sorted_idx]

    # ------------------------------------------------------------------
    # 7) Keep only first k PCs
    # ------------------------------------------------------------------
    if k is not None:
        eigenvectors = eigenvectors[:, :k]

    # ------------------------------------------------------------------
    # 8) Project data
    # ------------------------------------------------------------------
    X_pca = X_hat @ eigenvectors

    return X_pca, X_mean, X_std, eigenvalues, eigenvectors


def apply_pca(X_new, pca_eigenvectors, original_mean, original_std=None):
    """
    Apply PCA to new data using previously computed eigenvectors (principal components) and the original mean & std.

    Args:
        X_new (torch.Tensor): New input tensor of shape (n, d), where n is the number of samples and d is the feature dimension.
        pca_eigenvectors (torch.Tensor): Eigenvectors (principal components) from the previously computed PCA, of shape (d, k), where k is the number of principal components.
        original_mean (torch.Tensor): The mean of the original data (used when computing PCA), of shape (d,).
        original_std (torch.Tensor): The standard deviation of the original data (used when computing PCA), of shape (d,).

    Returns:
        torch.Tensor: Transformed data of shape (n, k) projected onto the principal components.
    """
    # Step 1: Normalize the new data using the original mean & std
    X_new_hat = X_new - original_mean
    if original_std is not None:
        X_new_hat = X_new_hat / original_std

    # Step 2: Project the new data onto the principal components
    X_new_pca = torch.mm(X_new_hat, pca_eigenvectors)

    return X_new_pca


def tensor_to_rgb(tensor: torch.Tensor, tensor_min=None, tensor_max=None):
    # Ensure the tensor is of shape (n, 3)
    if tensor.shape[-1] != 3:
        raise ValueError("Input tensor must have shape (n, 3)")

    if tensor_min is None:
        tensor_min = tensor.min()

    # Normalize the tensor values to the range [0, 255]
    tensor = tensor - tensor_min
    if tensor_max is None:
        tensor_max = tensor.max()
    tensor = tensor / (tensor_max + 1e-8)
    tensor = tensor * 255.0
    tensor = tensor.clamp(0, 255)

    # Convert to integer type
    tensor = tensor.to(torch.uint8)

    return tensor, tensor_min, tensor_max
