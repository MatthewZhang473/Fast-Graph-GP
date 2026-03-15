import torch

from .base import BaseGRFKernel
from .low_rank import LowRankGRFKernel


class GRFGeneralKernel(BaseGRFKernel):
    """
    Learnable GRF kernel with a unconstrained modulation function.
    """

    def __init__(self, rw_mats, max_walk_length: int, **kwargs):
        super().__init__(rw_mats=rw_mats, **kwargs)
        self.max_walk_length = max_walk_length
        self.register_parameter(
            name="raw_modulation_function",
            parameter=torch.nn.Parameter(torch.randn(max_walk_length)),
        )

    @property
    def modulation_function(self) -> torch.Tensor:
        return self.raw_modulation_function


class LowRankGeneralGRFKernel(LowRankGRFKernel):
    def __init__(
        self, rw_mats, max_walk_length: int, proj_dim: int, jlt_seed: int = 42, **kwargs
    ):
        super().__init__(
            rw_mats=rw_mats, proj_dim=proj_dim, jlt_seed=jlt_seed, **kwargs
        )
        self.register_parameter(
            name="raw_modulation_function",
            parameter=torch.nn.Parameter(torch.randn(max_walk_length)),
        )

    @property
    def modulation_function(self) -> torch.Tensor:
        return self.raw_modulation_function
