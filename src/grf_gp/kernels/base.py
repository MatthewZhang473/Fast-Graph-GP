import torch
import gpytorch
from abc import ABC, abstractmethod
from typing import Any, cast

from grf_gp._types import (
    DenseMatrix,
    FeatureMatrixLike,
    NodeIndexTensor,
    RandomWalkMatrices,
)


class BaseGRFKernel(gpytorch.kernels.Kernel, ABC):
    rw_mats: RandomWalkMatrices

    def __init__(self, rw_mats: RandomWalkMatrices, **kwargs) -> None:
        super().__init__(**kwargs)
        self.rw_mats = rw_mats

    @property
    @abstractmethod
    def modulation_function(self) -> torch.Tensor:
        pass

    def forward(
        self,
        x1_idx: NodeIndexTensor | None = None,
        x2_idx: NodeIndexTensor | None = None,
        diag: bool = False,
        **params,
    ) -> FeatureMatrixLike:
        """
        Efficient Implementation of K[x1, x2], where K = Phi @ Phi^T
        """

        phi = self._get_feature_matrix()
        phi_indexable = cast(Any, phi)

        # Handle indexing
        if x1_idx is not None:
            x1_idx = x1_idx.long().flatten()
            phi_x1 = cast(Any, phi_indexable)[x1_idx]
        else:
            phi_x1 = phi
        if x2_idx is not None:
            x2_idx = x2_idx.long().flatten()
            phi_x2 = cast(Any, phi_indexable)[x2_idx]
        else:
            phi_x2 = phi

        if diag:
            # diag(A @ B^T) = sum(A * B, dim=-1)
            return (phi_x1 * phi_x2).sum(dim=-1)

        else:
            # Return K[x1, x2] = Phi[x1, :] @ Phi[x2, :]^T
            return phi_x1 @ phi_x2.transpose(-1, -2)

    def _get_feature_matrix(self) -> FeatureMatrixLike:
        """
        Returns the feature matrix Phi,
        the ith row is the GRF vector for the ith node.
        Ideally this should be lazy-evaluated linear operator.
        """
        return cast(
            FeatureMatrixLike,
            sum(
                mod_vec * mat
                for mod_vec, mat in zip(
                    self.modulation_function,
                    self.rw_mats,
                )
            ),
        )


class BaseExactKernel(gpytorch.kernels.Kernel, ABC):
    """
    Base class for exact graph kernels defined via a full kernel matrix.
    """

    @abstractmethod
    def _full_kernel_matrix(self) -> DenseMatrix:
        pass

    def forward(
        self,
        x1: NodeIndexTensor,
        x2: NodeIndexTensor,
        diag: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        K_full = self._full_kernel_matrix()
        i1 = x1.long().flatten()
        i2 = x2.long().flatten()
        if diag:
            return K_full[i1, i1]
        return K_full[i1][:, i2]
