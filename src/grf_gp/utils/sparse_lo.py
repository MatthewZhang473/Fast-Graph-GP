from linear_operator.operators import LinearOperator
import torch


class SparseLinearOperator(LinearOperator):
    """
    A LinearOperator that wraps a sparse CSR tensor and performs
    sparse matrix @ dense tensor operations efficiently.
    """

    sparse_csr_tensor: torch.Tensor

    def __init__(self, sparse_csr_tensor: torch.Tensor) -> None:
        if not sparse_csr_tensor.is_sparse_csr:
            raise ValueError("Input tensor must be a sparse CSR tensor")
        self.sparse_csr_tensor = sparse_csr_tensor
        super().__init__(sparse_csr_tensor)

    def _matmul(self, rhs: torch.Tensor) -> torch.Tensor:
        return self.sparse_csr_tensor.matmul(rhs)

    def _size(self) -> torch.Size:
        return self.sparse_csr_tensor.size()

    def _transpose_nonbatch(self) -> "SparseLinearOperator":
        """Tranpose the linear operator by converting:
        CSR tensor -> CSC tensor -> CSR tensor."""
        return SparseLinearOperator(self.sparse_csr_tensor.t().to_sparse_csr())
