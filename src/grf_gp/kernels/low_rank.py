import torch
from abc import ABC
from typing import cast

from .base import BaseGRFKernel
from grf_gp._types import DenseMatrix, FeatureMatrixLike, RandomWalkMatrices


class LowRankGRFKernel(BaseGRFKernel, ABC):
    proj_dim: int
    jlt_proj: torch.Tensor

    def __init__(
        self, rw_mats: RandomWalkMatrices, proj_dim: int, jlt_seed: int = 42, **kwargs
    ) -> None:
        super().__init__(rw_mats=rw_mats, **kwargs)
        # TODO: check the proj_dim is big enough
        self.proj_dim = proj_dim
        first_rw_mat = rw_mats[0]
        full_dim = cast(
            int, first_rw_mat.shape[-1]
        )  # a.k.a. number of nodes in the graph
        device = first_rw_mat.device

        # Initialize the JLT projection matrix
        gen = torch.Generator(device=device).manual_seed(jlt_seed)
        jlt_proj = torch.randn(full_dim, proj_dim, generator=gen, device=device)
        jlt_proj = jlt_proj * (proj_dim**-0.5)

        self.register_buffer("jlt_proj", jlt_proj)

    def _get_feature_matrix(self) -> DenseMatrix:
        """
        Calculates Phi_low_rank = Phi_full @ JLT_proj
        """
        phi_full = cast(FeatureMatrixLike, super()._get_feature_matrix())
        # This final sparse @ dense operation return a dense tensor
        # Howeverm given nnz(Phi) = O(N), the time complexity is O(N * D_proj)
        return cast(DenseMatrix, phi_full @ self.jlt_proj)
