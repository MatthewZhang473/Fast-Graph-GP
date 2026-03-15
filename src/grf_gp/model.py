import torch
import gpytorch
from abc import ABC, abstractmethod
from linear_operator.operators import DiagLinearOperator
from .kernels.base import BaseGRFKernel, BaseExactKernel
from .kernels.low_rank import LowRankGRFKernel
from .inference import pathwise_conditioning, woodbury_pathwise_conditioning


class GraphGP(gpytorch.models.ExactGP, ABC):
    def __init__(
        self,
        x_train,
        y_train,
        likelihood,
        kernel: BaseGRFKernel | BaseExactKernel,
    ):
        super().__init__(x_train, y_train, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = kernel
        self.x_train = x_train
        self.y_train = y_train

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    @abstractmethod
    def predict(self, x_test, **kwargs):
        pass


class GRFGP(GraphGP):
    def predict_sample(self, x_test, n_samples=64):
        with torch.no_grad():
            x_train = self.x_train.int().flatten()
            x_test = x_test.int().flatten()
            phi = self.covar_module._get_feature_matrix()
            noise_std = self.likelihood.noise.to(
                device=x_test.device, dtype=phi.dtype
            ).sqrt()
            return pathwise_conditioning(
                x_train=x_train,
                x_test=x_test,
                phi=phi,
                y_train=self.y_train,
                noise_std=noise_std,
                batch_size=n_samples,
                device=x_test.device,
            )

    def predict(self, x_test, batch_size=64):
        samples = self.predict_sample(x_test=x_test, n_samples=batch_size)
        # Estimate posterior mean and variance from samples
        mean = samples.mean(dim=0)
        var = samples.var(dim=0, unbiased=True)
        covar = DiagLinearOperator(var)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class LowRankGRFGP(GRFGP):
    def __init__(self, x_train, y_train, likelihood, kernel: LowRankGRFKernel):
        super().__init__(x_train, y_train, likelihood, kernel)
        if not isinstance(self.covar_module, LowRankGRFKernel):
            raise TypeError("LowRankGRFGP requires a LowRankGRFKernel.")

    def predict_sample(self, x_test, n_samples=64):
        with torch.no_grad():
            x_train = self.x_train.int().flatten()
            x_test = x_test.int().flatten()
            phi = self.covar_module._get_feature_matrix()
            noise_std = self.likelihood.noise.to(
                device=x_test.device, dtype=phi.dtype
            ).sqrt()
            return woodbury_pathwise_conditioning(
                x_train=x_train,
                x_test=x_test,
                phi=phi,
                y_train=self.y_train,
                noise_std=noise_std,
                batch_size=n_samples,
                device=x_test.device,
            )


class ExactGraphGP(GraphGP):
    def predict(self, x_test, **kwargs):
        self.eval()
        self.likelihood.eval()
        with torch.no_grad():
            return self.likelihood(self(x_test))
