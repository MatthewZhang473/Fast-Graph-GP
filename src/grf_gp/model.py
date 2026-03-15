import torch
import gpytorch
from abc import ABC, abstractmethod
from typing import cast
from linear_operator.operators import DiagLinearOperator
from .kernels.base import BaseGRFKernel, BaseExactKernel
from .kernels.low_rank import LowRankGRFKernel
from .inference import pathwise_conditioning, woodbury_pathwise_conditioning
from ._types import NodeIndexTensor, ObservationTensor


class GraphGP(gpytorch.models.ExactGP, ABC):
    x_train: NodeIndexTensor
    y_train: ObservationTensor

    def __init__(
        self,
        x_train: NodeIndexTensor,
        y_train: ObservationTensor,
        likelihood,
        kernel: BaseGRFKernel | BaseExactKernel,
    ) -> None:
        super().__init__(x_train, y_train, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = kernel
        self.x_train = x_train
        self.y_train = y_train

    def forward(self, x: NodeIndexTensor) -> gpytorch.distributions.MultivariateNormal:
        mean_x = cast(torch.Tensor, self.mean_module(x))
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    @abstractmethod
    def predict(
        self, x_test: NodeIndexTensor, **kwargs
    ) -> gpytorch.distributions.MultivariateNormal:
        pass


class GRFGP(GraphGP):
    def predict_sample(
        self, x_test: NodeIndexTensor, n_samples: int = 64
    ) -> torch.Tensor:
        with torch.no_grad():
            x_train = self.x_train.int().flatten()
            x_test = x_test.int().flatten()
            covar_module = cast(BaseGRFKernel, self.covar_module)
            phi = covar_module._get_feature_matrix()
            likelihood = cast(gpytorch.likelihoods.GaussianLikelihood, self.likelihood)
            noise_std = likelihood.noise.to(
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

    def predict(
        self, x_test: NodeIndexTensor, batch_size: int = 64
    ) -> gpytorch.distributions.MultivariateNormal:
        samples = self.predict_sample(x_test=x_test, n_samples=batch_size)
        # Estimate posterior mean and variance from samples
        mean = samples.mean(dim=0)
        var = samples.var(dim=0, unbiased=True)
        covar = DiagLinearOperator(var)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class LowRankGRFGP(GRFGP):
    def __init__(
        self,
        x_train: NodeIndexTensor,
        y_train: ObservationTensor,
        likelihood,
        kernel: LowRankGRFKernel,
    ) -> None:
        super().__init__(x_train, y_train, likelihood, kernel)
        if not isinstance(self.covar_module, LowRankGRFKernel):
            raise TypeError("LowRankGRFGP requires a LowRankGRFKernel.")

    def predict_sample(
        self, x_test: NodeIndexTensor, n_samples: int = 64
    ) -> torch.Tensor:
        with torch.no_grad():
            x_train = self.x_train.int().flatten()
            x_test = x_test.int().flatten()
            covar_module = cast(LowRankGRFKernel, self.covar_module)
            phi = covar_module._get_feature_matrix()
            likelihood = cast(gpytorch.likelihoods.GaussianLikelihood, self.likelihood)
            noise_std = likelihood.noise.to(
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
    def predict(
        self, x_test: NodeIndexTensor, **kwargs
    ) -> gpytorch.distributions.MultivariateNormal:
        self.eval()
        likelihood = cast(gpytorch.likelihoods.Likelihood, self.likelihood)
        likelihood.eval()
        with torch.no_grad():
            return cast(
                gpytorch.distributions.MultivariateNormal, likelihood(self(x_test))
            )
