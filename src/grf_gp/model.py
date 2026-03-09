import torch
import gpytorch
from abc import ABC, abstractmethod
from linear_operator.operators import DiagLinearOperator
from .kernels.base import BaseGRFKernel
from .inference import pathwise_conditioning


class GraphGP(gpytorch.models.ExactGP, ABC):
    def __init__(self, x_train, y_train, likelihood, kernel: BaseGRFKernel):
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


class GraphGRFGP(GraphGP):
    def predict_sample(self, x_test, n_samples=64):
        with torch.no_grad():
            x_train = self.x_train.int().flatten()
            x_test = x_test.int().flatten()
            phi = self.covar_module._get_feature_matrix()
            noise_std = torch.sqrt(
                torch.tensor(self.likelihood.noise.item(), device=x_test.device)
            )
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


class GraphExactGP(GraphGP):
    def predict(self, x_test, **kwargs):
        self.eval()
        self.likelihood.eval()
        with torch.no_grad():
            return self.likelihood(self(x_test))
