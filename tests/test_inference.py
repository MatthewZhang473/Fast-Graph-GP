import torch

from grf_gp.inference import pathwise_conditioning, woodbury_pathwise_conditioning
from grf_gp.utils.sparse_lo import SparseLinearOperator


def test_pathwise_conditioning_smoke_shape():
    eye = torch.eye(4, dtype=torch.float64)
    phi = 1 * SparseLinearOperator(eye.to_sparse_csr())
    x_train = torch.tensor([0, 1], dtype=torch.int64)
    x_test = torch.tensor([2, 3], dtype=torch.int64)
    y_train = torch.tensor([0.5, -0.5], dtype=torch.float64)
    noise_std = torch.tensor(0.1, dtype=torch.float64)

    samples = pathwise_conditioning(
        x_train=x_train,
        x_test=x_test,
        phi=phi,
        y_train=y_train,
        noise_std=noise_std,
        batch_size=5,
        device=torch.device("cpu"),
    )

    assert samples.shape == (5, 2)


def test_woodbury_pathwise_conditioning_smoke_shape():
    phi_dense = torch.tensor(
        [
            [1.0, 0.0],
            [0.5, -0.5],
        ],
        dtype=torch.float64,
    )

    x_train = torch.tensor([0], dtype=torch.int64)
    x_test = torch.tensor([1], dtype=torch.int64)
    y_train = torch.tensor([0.5], dtype=torch.float64)
    noise_std = torch.tensor(0.1, dtype=torch.float64)

    samples = woodbury_pathwise_conditioning(
        x_train=x_train,
        x_test=x_test,
        phi=phi_dense,
        y_train=y_train,
        noise_std=noise_std,
        batch_size=1,
        device=torch.device("cpu"),
    )

    assert samples.dtype == torch.float64
    assert samples.shape == (1, 1)
