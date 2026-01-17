import torch
from gpytorch import settings as gsettings
from linear_operator.operators import IdentityLinearOperator
from linear_operator.utils import linear_cg
from utils.linear_operator import SparseLinearOperator


def pathwise_conditioning(
    x_train,
    x_test,
    phi,
    y_train,
    noise_std,
    batch_size,
    device,
):
    """Pathwise conditioning to sample from the posterior."""
    phi_train = phi[x_train, :]
    phi_test = phi[x_test, :]

    K_train_train = phi_train @ phi_train.T
    K_test_train = phi_test @ phi_train.T
    noise_variance = noise_std.pow(2)
    A = K_train_train + noise_variance * IdentityLinearOperator(
        phi_train.size(0), device=device
    )
    print(A)

    eps_prior = torch.randn(batch_size, phi.size(0), device=device)
    eps_obs = noise_std * torch.randn(batch_size, phi_train.size(0), device=device)

    f_test_prior = eps_prior @ phi_test.T
    f_train_prior = eps_prior @ phi_train.T

    residual = y_train.unsqueeze(0) - (f_train_prior + eps_obs)
    v = linear_cg(A._matmul, residual.T, tolerance=gsettings.cg_tolerance.value())

    return f_test_prior + (K_test_train @ v).T


if __name__ == "__main__":
    # Minimal example usage
    device = torch.device("cpu")
    crow_indices = torch.tensor([0, 2, 4, 6], device=device)
    col_indices = torch.tensor([0, 2, 1, 2, 0, 1], device=device)
    values = torch.tensor([1.0, 0.5, 0.3, 1.0, 1.0, 1.0], device=device)
    phi_csr = torch.sparse_csr_tensor(crow_indices, col_indices, values, size=(3, 3))
    phi = 1 * SparseLinearOperator(phi_csr)
    print(phi)

    x_train = torch.tensor([0, 1], dtype=torch.int64, device=device)
    x_test = torch.tensor([2], dtype=torch.int64, device=device)
    y_train = torch.tensor([1.0, -1.0], device=device)
    noise_std = torch.tensor(0.1, device=device)

    samples = pathwise_conditioning(
        x_train=x_train,
        x_test=x_test,
        phi=phi,
        y_train=y_train,
        noise_std=noise_std,
        batch_size=4,
        device=device,
    )
    print("Posterior samples shape:", samples.shape)
    print("Posterior samples:", samples)
