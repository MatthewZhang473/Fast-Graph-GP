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

    :param x_train: Training node indices.
    :param x_test: Test node indices.
    :param phi: Feature matrix whose rows correspond to graph nodes.
    :param y_train: Training targets.
    :param noise_std: Observation noise standard deviation.
    :param batch_size: Number of posterior samples to draw.
    :param device: Device on which the computation is performed.
    :returns: Posterior function samples at ``x_test`` with shape
        ``(batch_size, len(x_test))``.
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


def woodbury_pathwise_conditioning(
    x_train,
    x_test,
    phi,
    y_train,
    noise_std,
    batch_size,
    device,
):
    r"""
    Perform pathwise conditioning using a Woodbury solve in feature space.

    This is the low-rank analogue of ``pathwise_conditioning`` for kernels of the
    form :math:`\hat{\mathbf{K}} = \mathbf{\Phi}\mathbf{\Phi}^\top`, where
    :math:`\mathbf{\Phi}` has a small feature dimension. It applies the Woodbury
    identity to solve an equivalent system in small feature space.

    The update follows the identity:

    .. math::

        \left(
            \mathbf{\Phi}_{\mathbf{x}}\mathbf{\Phi}_{\mathbf{x}}^\top
            + \sigma_n^2 \mathbf{I}
        \right)^{-1}
        =
        \sigma_n^{-2}\mathbf{I}
        -
        \sigma_n^{-4}\mathbf{\Phi}_{\mathbf{x}}
        \left(
            \mathbf{I}
            + \sigma_n^{-2}
            \mathbf{\Phi}_{\mathbf{x}}^\top \mathbf{\Phi}_{\mathbf{x}}
        \right)^{-1}
        \mathbf{\Phi}_{\mathbf{x}}^\top

    :param x_train: Training node indices.
    :param x_test: Test node indices.
    :param phi: Low-rank feature matrix whose rows correspond to graph nodes.
    :param y_train: Training targets.
    :param noise_std: Observation noise standard deviation.
    :param batch_size: Number of posterior samples to draw.
    :param device: Device on which the computation is performed.
    :returns: Posterior function samples at ``x_test`` with shape
        ``(batch_size, len(x_test))``.
    """
    phi_train = phi[x_train, :]
    phi_test = phi[x_test, :]
    dtype = phi.dtype

    noise_variance = noise_std.pow(2)
    proj_dim = phi_train.size(1)

    eps_prior = torch.randn(batch_size, proj_dim, device=device, dtype=dtype)
    eps_obs = noise_std * torch.randn(
        batch_size, phi_train.size(0), device=device, dtype=dtype
    )

    f_test_prior = eps_prior @ phi_test.T
    f_train_prior = eps_prior @ phi_train.T

    residual = y_train.unsqueeze(0) - (f_train_prior + eps_obs)
    gram_features = phi_train.T @ phi_train
    woodbury_system = torch.eye(proj_dim, device=device, dtype=dtype)
    woodbury_system += gram_features / noise_variance
    rhs = phi_train.T @ residual.T
    u = torch.linalg.solve(woodbury_system, rhs)

    return f_test_prior + (phi_test @ (u / noise_variance)).T
