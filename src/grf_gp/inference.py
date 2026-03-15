import torch
from gpytorch import settings as gsettings
from linear_operator.operators import IdentityLinearOperator
from linear_operator.utils import linear_cg


def pathwise_conditioning(
    x_train,
    x_test,
    phi,
    y_train,
    noise_std,
    batch_size,
    device,
):
    r"""
    Perform pathwise conditioning to sample from the Gaussian Process posterior.

    This function implements the Matheron's rule (or pathwise update) to produce
    exact posterior samples by perturbing prior samples with a correction term
    derived from the residuals of the observations.

    The update follows the identity:

    .. math::

        \mathbf{g}_{|\mathbf{y}}(\cdot) =
        \mathbf{g}(\cdot)
        + \hat{\mathbf{K}}_{(\cdot)\mathbf{x}}
        \left(
            \hat{\mathbf{K}}_{\mathbf{x}\mathbf{x}} + \sigma_n^2 \mathbf{I}
        \right)^{-1}
        \left(
            \mathbf{y} - (\mathbf{g}(\mathbf{x}) + \boldsymbol{\varepsilon})
        \right)
    """
    phi_train = phi[x_train, :]
    phi_test = phi[x_test, :]
    dtype = phi.dtype

    K_train_train = phi_train @ phi_train.T
    K_test_train = phi_test @ phi_train.T
    noise_variance = noise_std.pow(2)
    A = K_train_train + noise_variance * IdentityLinearOperator(
        phi_train.size(0), device=device
    )

    eps_prior = torch.randn(batch_size, phi.size(1), device=device, dtype=dtype)
    eps_obs = noise_std * torch.randn(
        batch_size, phi_train.size(0), device=device, dtype=dtype
    )

    f_test_prior = eps_prior @ phi_test.T
    f_train_prior = eps_prior @ phi_train.T

    residual = y_train.unsqueeze(0) - (f_train_prior + eps_obs)
    v = linear_cg(A._matmul, residual.T, tolerance=gsettings.cg_tolerance.value())

    return f_test_prior + (K_test_train @ v).T
