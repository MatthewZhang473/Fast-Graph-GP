import torch
import gpytorch
from abc import ABC, abstractmethod
from linear_operator.operators import DiagLinearOperator
from .kernels.base import BaseGRFKernel, BaseExactKernel
from .kernels.low_rank import LowRankGRFKernel
from .inference import pathwise_conditioning, woodbury_pathwise_conditioning


class GraphGP(gpytorch.models.ExactGP, ABC):
    """Abstract base class for graph Gaussian process models."""

    def __init__(
        self,
        x_train,
        y_train,
        likelihood,
        kernel: BaseGRFKernel | BaseExactKernel,
    ):
        """Initialize the graph GP model.

        :param x_train: Training node indices.
        :param y_train: Training targets.
        :param likelihood: Likelihood module used by the GP model.
        :param kernel: Graph kernel defining the covariance structure.
        :returns: ``None``.
        """
        super().__init__(x_train, y_train, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = kernel
        self.x_train = x_train
        self.y_train = y_train

    def forward(self, x):
        """Construct the GP prior at the requested inputs.

        :param x: Input node indices.
        :returns: Multivariate normal prior distribution at ``x``.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    @abstractmethod
    def predict(self, x_test, **kwargs):
        """Predict at test inputs.

        :param x_test: Test node indices.
        :param kwargs: Model-specific prediction options.
        :returns: Predictive distribution or samples at ``x_test``.
        """
        pass


class GRFGP(GraphGP):
    """Graph GP model that predicts by pathwise posterior sampling."""

    def predict_sample(self, x_test, n_samples=64):
        """Draw posterior samples at test nodes.

        :param x_test: Test node indices.
        :param n_samples: Number of posterior samples to draw.
        :returns: Posterior samples with shape ``(n_samples, len(x_test))``.
        """
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
        """Approximate the predictive distribution from posterior samples.

        :param x_test: Test node indices.
        :param batch_size: Number of posterior samples used for the estimate.
        :returns: Approximate predictive multivariate normal distribution.
        """
        samples = self.predict_sample(x_test=x_test, n_samples=batch_size)
        # Estimate posterior mean and variance from samples
        mean = samples.mean(dim=0)
        var = samples.var(dim=0, unbiased=True)
        covar = DiagLinearOperator(var)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class LowRankGRFGP(GRFGP):
    """Low-rank graph GP model using Woodbury pathwise conditioning."""

    def __init__(self, x_train, y_train, likelihood, kernel: LowRankGRFKernel):
        """Initialize the low-rank graph GP model.

        :param x_train: Training node indices.
        :param y_train: Training targets.
        :param likelihood: Likelihood module used by the GP model.
        :param kernel: Low-rank graph kernel defining the covariance structure.
        :returns: ``None``.
        :raises TypeError: If ``kernel`` is not a :class:`LowRankGRFKernel`.
        """
        super().__init__(x_train, y_train, likelihood, kernel)
        if not isinstance(self.covar_module, LowRankGRFKernel):
            raise TypeError("LowRankGRFGP requires a LowRankGRFKernel.")

    def predict_sample(self, x_test, n_samples=64):
        """Draw posterior samples using the low-rank feature representation.

        :param x_test: Test node indices.
        :param n_samples: Number of posterior samples to draw.
        :returns: Posterior samples with shape ``(n_samples, len(x_test))``.
        """
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
    """Exact graph GP model using the full kernel matrix."""

    def predict(self, x_test, **kwargs):
        """Evaluate the exact predictive distribution.

        :param x_test: Test node indices.
        :param kwargs: Unused prediction options.
        :returns: Predictive distribution evaluated at ``x_test``.
        """
        self.eval()
        self.likelihood.eval()
        with torch.no_grad():
            return self.likelihood(self(x_test))
