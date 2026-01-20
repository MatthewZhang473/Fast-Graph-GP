import torch

from grf_gp.utils.sparse_lo import SparseLinearOperator
from linear_operator.utils.linear_cg import linear_cg


def test_sparse_linear_operator_matmul_and_transpose():
    indices = torch.tensor([[0, 0, 1, 2], [0, 2, 1, 0]])
    values = torch.tensor([1.0, 2.0, 3.0, 4.0])
    size = (3, 3)

    sparse_csr = torch.sparse_coo_tensor(indices, values, size).to_sparse_csr()
    op = SparseLinearOperator(sparse_csr)

    rhs = torch.tensor([[1.0], [2.0], [3.0]])
    result = op @ rhs
    expected = sparse_csr.to_dense() @ rhs
    assert torch.allclose(result, expected)

    lhs = torch.tensor([[1.0, 2.0, 3.0]])
    result_t = lhs @ op
    expected_t = lhs @ sparse_csr.to_dense()
    assert torch.allclose(result_t, expected_t)


def test_sparse_linear_operator_linear_cg():
    indices = torch.tensor([[0, 1, 1, 2], [0, 0, 1, 2]])
    values = torch.tensor([4.0, 1.0, 4.0, 4.0])
    size = (3, 3)
    sparse_csr = torch.sparse_coo_tensor(indices, values, size).to_sparse_csr()
    M = SparseLinearOperator(sparse_csr)
    A = M @ M.T + torch.eye(3)
    b = torch.tensor([1.0, 2.0, 3.0])
    x = linear_cg(A.matmul, b, max_iter=100)
    result = A.matmul(x)
    assert torch.allclose(result, b, atol=1e-5)


def test_sparse_linear_operator_indexing_and_matmul():
    """Test indexing a SparseLinearOperator
    and then performing matrix multiplication.
    it would fail for now."""
    device = torch.device("cpu")
    crow_indices = torch.tensor([0, 2, 4, 6], device=device)
    col_indices = torch.tensor([0, 2, 1, 2, 0, 1], device=device)
    values = torch.tensor([1.0, 0.5, 0.3, 1.0, 1.0, 1.0], device=device)
    csr = torch.sparse_csr_tensor(crow_indices, col_indices, values, size=(3, 3))
    M = SparseLinearOperator(csr)

    indices = torch.tensor([0, 1], device=device)
    M_indexed = M[indices, :]
    phi = M_indexed @ M_indexed.T

    # Expected result: compute manually from dense representation
    M_dense = csr.to_dense()
    M_indexed_dense = M_dense[indices, :]
    expected = M_indexed_dense @ M_indexed_dense.T

    assert torch.allclose(
        phi.to_dense() if hasattr(phi, "to_dense") else phi, expected, atol=1e-5
    )
