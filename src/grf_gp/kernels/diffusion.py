import torch
import torch.nn.functional as F
from typing import cast
from .base import BaseExactKernel, BaseGRFKernel
from .low_rank import LowRankGRFKernel
from grf_gp._types import DenseMatrix, RandomWalkMatrices


def diffusion_formula(length: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    r"""
    Compute the diffusion modulation term

    .. math::
    f(\ell, \beta) = \frac{(-\beta)^\ell}{2^\ell \, \Gamma(\ell + 1)}

    where :math:`\ell` is the walk length and :math:`\beta` the diffusion rate.
    """
    length = length.to(dtype=beta.dtype, device=beta.device)
    numerator = torch.pow(-beta, length)
    denominator = torch.pow(
        torch.tensor(2.0, dtype=beta.dtype, device=beta.device), length
    )
    denominator *= torch.exp(torch.lgamma(length + 1.0))
    return numerator / denominator


class DiffusionModule:
    """Shared logic for diffusion parameters and modulation."""

    raw_beta: torch.nn.Parameter
    raw_sigma_f: torch.nn.Parameter

    def _init_diffusion_params(self) -> None:
        module = cast(torch.nn.Module, self)
        module.register_parameter("raw_beta", torch.nn.Parameter(torch.tensor(1.0)))
        module.register_parameter("raw_sigma_f", torch.nn.Parameter(torch.tensor(1.0)))

    @property
    def beta(self) -> torch.Tensor:
        return F.softplus(self.raw_beta)

    @property
    def sigma_f(self) -> torch.Tensor:
        return F.softplus(self.raw_sigma_f)

    def compute_modulation(self, max_walk_length: int) -> torch.Tensor:
        walk_lengths = torch.arange(
            max_walk_length, device=self.raw_beta.device, dtype=self.raw_beta.dtype
        )
        return self.sigma_f * diffusion_formula(walk_lengths, self.beta)


class DiffusionGRFKernel(BaseGRFKernel, DiffusionModule):
    def __init__(
        self, rw_mats: RandomWalkMatrices, max_walk_length: int, **kwargs
    ) -> None:
        super().__init__(rw_mats=rw_mats, **kwargs)
        self.max_walk_length = max_walk_length
        self._init_diffusion_params()

    @property
    def modulation_function(self):
        return self.compute_modulation(self.max_walk_length)


class DiffusionLowRankGRFKernel(LowRankGRFKernel, DiffusionModule):
    def __init__(
        self,
        rw_mats: RandomWalkMatrices,
        max_walk_length: int,
        proj_dim: int,
        jlt_seed: int = 42,
        **kwargs,
    ) -> None:
        super().__init__(
            rw_mats=rw_mats, proj_dim=proj_dim, jlt_seed=jlt_seed, **kwargs
        )
        self.max_walk_length = max_walk_length
        self._init_diffusion_params()

    @property
    def modulation_function(self):
        return self.compute_modulation(self.max_walk_length)


class DiffusionExactKernel(BaseExactKernel, DiffusionModule):
    def __init__(self, L: DenseMatrix, **kwargs) -> None:
        super().__init__(**kwargs)
        self.register_buffer("L", L)
        self._init_diffusion_params()

    def _full_kernel_matrix(self) -> DenseMatrix:
        laplacian = cast(torch.Tensor, self.L)
        return cast(
            DenseMatrix, self.sigma_f**2 * torch.matrix_exp(-self.beta * laplacian)
        )
