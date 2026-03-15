import torch
from abc import ABC

from .base import BaseGRFKernel


class LowRankGRFKernel(BaseGRFKernel, ABC):
    """Base class for low-rank GRF kernels using JLT projections."""

    def __init__(self, rw_mats, proj_dim: int, jlt_seed: int = 42, **kwargs):
        """Initialize the low-rank GRF kernel.

        :param rw_mats: Random-walk matrices used to construct the feature map.
        :param proj_dim: Johnson-Lindenstrauss projection dimension.
        :param jlt_seed: Random seed used for the projection matrix.
        :param kwargs: Additional keyword arguments passed to the base kernel.
        :returns: ``None``.
        """
        super().__init__(rw_mats=rw_mats, **kwargs)
        # TODO: check the proj_dim is big enough
        self.proj_dim = proj_dim
        full_dim = rw_mats[0].size(-1)  # a.k.a. number of nodes in the graph
        device = rw_mats[0].device

        # Initialize the JLT projection matrix
        gen = torch.Generator(device=device).manual_seed(jlt_seed)
        jlt_proj = torch.randn(full_dim, proj_dim, generator=gen, device=device)
        jlt_proj = jlt_proj * (proj_dim**-0.5)

        self.register_buffer("jlt_proj", jlt_proj)

    def _get_feature_matrix(self) -> torch.Tensor:
        """Construct the projected feature matrix :math:`\Phi_{\mathrm{low\_rank}}`.

        This computes :math:`\Phi_{\mathrm{full}} J`, where :math:`J` is the
        Johnson-Lindenstrauss projection matrix.

        :returns: Low-rank feature matrix.
        """
        phi_full = super()._get_feature_matrix()
        # This final sparse @ dense operation return a dense tensor
        # Howeverm given nnz(Phi) = O(N), the time complexity is O(N * D_proj)
        return phi_full @ self.jlt_proj
