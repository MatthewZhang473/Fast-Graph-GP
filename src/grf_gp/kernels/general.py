import torch

from .base import BaseGRFKernel
from .low_rank import LowRankGRFKernel


class GeneralGRFKernel(BaseGRFKernel):
    """Learnable GRF kernel with an unconstrained modulation function."""

    def __init__(self, rw_mats, max_walk_length: int, **kwargs):
        """Initialize the general GRF kernel.

        :param rw_mats: Random-walk matrices used to construct the feature map.
        :param max_walk_length: Number of walk lengths to parameterize.
        :param kwargs: Additional keyword arguments passed to the base kernel.
        :returns: ``None``.
        """
        super().__init__(rw_mats=rw_mats, **kwargs)
        self.max_walk_length = max_walk_length
        self.register_parameter(
            name="raw_modulation_function",
            parameter=torch.nn.Parameter(torch.randn(max_walk_length)),
        )

    @property
    def modulation_function(self) -> torch.Tensor:
        """Return the learnable modulation coefficients.

        :returns: Unconstrained modulation vector indexed by walk length.
        """
        return self.raw_modulation_function


class GeneralLowRankGRFKernel(LowRankGRFKernel):
    """Low-rank GRF kernel with an unconstrained modulation function."""

    def __init__(
        self, rw_mats, max_walk_length: int, proj_dim: int, jlt_seed: int = 42, **kwargs
    ):
        """Initialize the low-rank general GRF kernel.

        :param rw_mats: Random-walk matrices used to construct the feature map.
        :param max_walk_length: Number of walk lengths to parameterize.
        :param proj_dim: Johnson-Lindenstrauss projection dimension.
        :param jlt_seed: Random seed used for the projection matrix.
        :param kwargs: Additional keyword arguments passed to the base kernel.
        :returns: ``None``.
        """
        super().__init__(
            rw_mats=rw_mats, proj_dim=proj_dim, jlt_seed=jlt_seed, **kwargs
        )
        self.register_parameter(
            name="raw_modulation_function",
            parameter=torch.nn.Parameter(torch.randn(max_walk_length)),
        )

    @property
    def modulation_function(self) -> torch.Tensor:
        """Return the learnable modulation coefficients.

        :returns: Unconstrained modulation vector indexed by walk length.
        """
        return self.raw_modulation_function
