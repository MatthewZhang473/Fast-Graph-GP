import torch
import gpytorch
from abc import ABC, abstractmethod


class BaseGRFKernel(gpytorch.kernels.Kernel, ABC):
    """Base class for GRF kernels defined through random-walk features."""

    def __init__(self, rw_mats, **kwargs):
        """Initialize the GRF kernel.

        :param rw_mats: Random-walk matrices used to construct the feature map.
        :param kwargs: Additional keyword arguments passed to the base kernel.
        :returns: ``None``.
        """
        super().__init__(**kwargs)
        self.rw_mats = rw_mats

    @property
    @abstractmethod
    def modulation_function(self) -> torch.Tensor:
        """Return the modulation coefficients applied to the walk matrices.

        :returns: Modulation vector indexed by walk length.
        """
        pass

    def forward(self, x1_idx=None, x2_idx=None, diag=False, **params):
        """Evaluate kernel entries using the GRF feature matrix.

        Efficient implementation of :math:`K[x_1, x_2]`, where
        :math:`K = \Phi \Phi^\top`.

        :param x1_idx: Row indices for the first argument.
        :param x2_idx: Row indices for the second argument.
        :param diag: Whether to return only the diagonal entries.
        :param params: Additional keyword arguments accepted by the kernel API.
        :returns: Kernel matrix block or diagonal extracted from :math:`K`.
        """

        phi = self._get_feature_matrix()

        # Handle indexing
        if x1_idx is not None:
            x1_idx = x1_idx.long().flatten()
            phi_x1 = phi[x1_idx]
        else:
            phi_x1 = phi
        if x2_idx is not None:
            x2_idx = x2_idx.long().flatten()
            phi_x2 = phi[x2_idx]
        else:
            phi_x2 = phi

        if diag:
            # diag(A @ B^T) = sum(A * B, dim=-1)
            return (phi_x1 * phi_x2).sum(dim=-1)

        else:
            # Return K[x1, x2] = Phi[x1, :] @ Phi[x2, :]^T
            return phi_x1 @ phi_x2.transpose(-1, -2)

    def _get_feature_matrix(self):
        """Construct the GRF feature matrix :math:`\Phi`.

        The ``i``-th row is the GRF feature vector for the ``i``-th node.
        Ideally this would be represented as a lazy linear operator.

        :returns: Feature matrix :math:`\Phi`.
        """
        phi = sum(
            mod_vec * mat
            for mod_vec, mat in zip(
                self.modulation_function,
                self.rw_mats,
            )
        )
        return phi


class BaseExactKernel(gpytorch.kernels.Kernel, ABC):
    """Base class for exact graph kernels defined by a full kernel matrix."""

    @abstractmethod
    def _full_kernel_matrix(self) -> torch.Tensor:
        """Return the full kernel matrix.

        :returns: Dense kernel matrix over all graph nodes.
        """
        pass

    def forward(self, x1, x2, diag=False, **kwargs):
        """Extract entries from the full kernel matrix.

        :param x1: Row indices for the first argument.
        :param x2: Row indices for the second argument.
        :param diag: Whether to return diagonal entries only.
        :param kwargs: Additional unused keyword arguments.
        :returns: Kernel matrix block or diagonal extracted from the full matrix.
        """
        del kwargs
        K_full = self._full_kernel_matrix()
        i1 = x1.long().flatten()
        i2 = x2.long().flatten()
        if diag:
            return K_full[i1, i1]
        return K_full[i1][:, i2]
