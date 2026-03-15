import torch
from typing import cast

from .base import BaseGRFKernel
from .low_rank import LowRankGRFKernel
from grf_gp._types import RandomWalkMatrices


class GeneralGRFKernel(BaseGRFKernel):
    """
    Learnable GRF kernel with a unconstrained modulation function.
    """

    def __init__(
        self, rw_mats: RandomWalkMatrices, max_walk_length: int, **kwargs
    ) -> None:
        super().__init__(rw_mats=rw_mats, **kwargs)
        self.max_walk_length = max_walk_length
        self.register_parameter(
            name="raw_modulation_function",
            parameter=torch.nn.Parameter(torch.randn(max_walk_length)),
        )

    @property
    def modulation_function(self) -> torch.Tensor:
        return cast(torch.Tensor, self.raw_modulation_function)


class GeneralLowRankGRFKernel(LowRankGRFKernel):
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
        self.register_parameter(
            name="raw_modulation_function",
            parameter=torch.nn.Parameter(torch.randn(max_walk_length)),
        )

    @property
    def modulation_function(self) -> torch.Tensor:
        return cast(torch.Tensor, self.raw_modulation_function)
