import numpy as np


def generate_noisy_samples(K, noise_std=0.1, seed=42):
    """
    Generate noisy samples from a Gaussian process.

    Parameters:
    K (ndarray): Covariance matrix of shape (n, n).
    noise_std (float): Standard deviation of the additive noise.

    Returns:
    ndarray: Noisy samples of shape (n, 1).
    """
    np.random.seed(seed)
    num_nodes = K.shape[0]
    L = np.linalg.cholesky(K + 1e-6 * np.eye(num_nodes))  # Cholesky decomposition
    true_samples = L @ np.random.normal(
        size=(num_nodes, 1)
    )  # Sample from Gaussian process
    noise = noise_std * np.random.randn(num_nodes, 1)  # Additive noise
    Y_noisy = true_samples + noise  # Noisy observations
    return Y_noisy
