import torch
import gpytorch
from gpytorch import settings as gsettings
from linear_operator.utils import linear_cg
from linear_operator.operators import IdentityLinearOperator
from .kernels.base import BaseGRFKernel


class GraphGP(gpytorch.models.ExactGP):
    def __init__(self, x_train, y_train, likelihood, kernel: BaseGRFKernel):
        super().__init__(x_train, y_train, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def predict(self, x_test, batch_size=64):
        x_train = self.x_train.int().flatten()
        x_test = x_test.int().flatten()
        phi = self.covar_module._get_feature_matrix()
        noise_std = torch.sqrt(
            torch.tensor(self.likelihood.noise.item(), device=x_test.device)
        )
        return self._pathwise_conditioning(
            x_train=x_train,
            x_test=x_test,
            phi=phi,
            y_train=self.y_train,
            noise_std=noise_std,
            batch_size=batch_size,
            device=x_test.device,
        )

    def _pathwise_conditioning(
        self,
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
        v = linear_cg(
            A._matmul, residual.T, tolerance=gsettings.cg_tolerance.value()
        )

        return f_test_prior + (K_test_train @ v).T
