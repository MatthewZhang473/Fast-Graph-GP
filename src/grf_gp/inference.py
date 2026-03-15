import torch
from typing import Any, cast
from gpytorch import settings as gsettings
from linear_operator.operators import IdentityLinearOperator
from linear_operator.utils import linear_cg

from ._types import (
    Device,
    FeatureMatrixLike,
    NodeIndexTensor,
    ObservationTensor,
    SampleTensor,
)


def pathwise_conditioning(
    x_train: NodeIndexTensor,
    x_test: NodeIndexTensor,
    phi: FeatureMatrixLike,
    y_train: ObservationTensor,
    noise_std: torch.Tensor | float,
    batch_size: int,
    device: Device,
) -> SampleTensor:
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
    x_train_idx = cast(torch.Tensor, x_train.long().flatten())
    x_test_idx = cast(torch.Tensor, x_test.long().flatten())
    phi_indexable = cast(Any, phi)
    phi_train = phi_indexable[x_train_idx, :]
    phi_test = phi_indexable[x_test_idx, :]
    dtype = phi.dtype
    noise_std_tensor = torch.as_tensor(noise_std, device=device, dtype=dtype)
    feature_dim = cast(int, phi.shape[-1])
    num_train = cast(int, phi_train.shape[0])

    K_train_train = phi_train @ phi_train.T
    K_test_train = phi_test @ phi_train.T
    noise_variance = noise_std_tensor.pow(2)
    A = K_train_train + noise_variance * IdentityLinearOperator(
        num_train, device=device
    )

    eps_prior = torch.randn(batch_size, feature_dim, device=device, dtype=dtype)
    eps_obs = noise_std_tensor * torch.randn(
        batch_size, num_train, device=device, dtype=dtype
    )

    f_test_prior = eps_prior @ phi_test.T
    f_train_prior = eps_prior @ phi_train.T

    residual = y_train.unsqueeze(0) - (f_train_prior + eps_obs)
    linear_op = cast(IdentityLinearOperator, A)
    v = linear_cg(
        linear_op._matmul, residual.T, tolerance=gsettings.cg_tolerance.value()
    )

    return f_test_prior + (K_test_train @ v).T


def woodbury_pathwise_conditioning(
    x_train: NodeIndexTensor,
    x_test: NodeIndexTensor,
    phi: torch.Tensor,
    y_train: ObservationTensor,
    noise_std: torch.Tensor | float,
    batch_size: int,
    device: Device,
) -> SampleTensor:
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
    """
    x_train_idx = cast(torch.Tensor, x_train.long().flatten())
    x_test_idx = cast(torch.Tensor, x_test.long().flatten())
    phi_train = phi[x_train_idx, :]
    phi_test = phi[x_test_idx, :]
    dtype = phi.dtype
    noise_std_tensor = torch.as_tensor(noise_std, device=device, dtype=dtype)

    noise_variance = noise_std_tensor.pow(2)
    proj_dim = cast(int, phi_train.shape[1])
    num_train = cast(int, phi_train.shape[0])

    eps_prior = torch.randn(batch_size, proj_dim, device=device, dtype=dtype)
    eps_obs = noise_std_tensor * torch.randn(
        batch_size, num_train, device=device, dtype=dtype
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
