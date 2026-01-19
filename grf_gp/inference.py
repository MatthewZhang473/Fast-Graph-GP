import torch
from gpytorch import settings as gsettings
from linear_operator.operators import IdentityLinearOperator
from linear_operator.utils import linear_cg
from grf_gp.utils.sparse_lo import SparseLinearOperator


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

    eps_prior = torch.randn(batch_size, phi.size(0), device=device)
    eps_obs = noise_std * torch.randn(batch_size, phi_train.size(0), device=device)

    f_test_prior = eps_prior @ phi_test.T
    f_train_prior = eps_prior @ phi_train.T

    residual = y_train.unsqueeze(0) - (f_train_prior + eps_obs)
    v = linear_cg(A._matmul, residual.T, tolerance=gsettings.cg_tolerance.value())

    return f_test_prior + (K_test_train @ v).T
