from linear_operator.operators import LinearOperator


class SparseLinearOperator(LinearOperator):
    """Wrap a sparse CSR tensor as a linear operator."""

    def __init__(self, sparse_csr_tensor):
        """Initialize the sparse linear operator.

        :param sparse_csr_tensor: Sparse CSR tensor backing the operator.
        :returns: ``None``.
        :raises ValueError: If ``sparse_csr_tensor`` is not in CSR format.
        """
        if not sparse_csr_tensor.is_sparse_csr:
            raise ValueError("Input tensor must be a sparse CSR tensor")
        self.sparse_csr_tensor = sparse_csr_tensor
        super().__init__(sparse_csr_tensor)

    def _matmul(self, rhs):
        """Multiply the operator by a dense right-hand side.

        :param rhs: Dense right-hand side tensor.
        :returns: Product of the sparse operator and ``rhs``.
        """
        return self.sparse_csr_tensor.matmul(rhs)

    def _size(self):
        """Return the operator shape.

        :returns: Shape of the wrapped sparse tensor.
        """
        return self.sparse_csr_tensor.size()

    def _transpose_nonbatch(self):
        """Transpose the non-batch dimensions of the operator.

        The transpose is computed by converting the wrapped CSR tensor through
        PyTorch's transpose operation and back to CSR format.

        :returns: Transposed sparse linear operator.
        """
        return SparseLinearOperator(self.sparse_csr_tensor.t().to_sparse_csr())
