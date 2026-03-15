import pytest
import torch

from grf_gp.kernels.base import BaseGRFKernel


class DummyGRFKernel(BaseGRFKernel):
    def __init__(self, rw_mats, modulation_function):
        super().__init__(rw_mats=rw_mats)
        self._modulation_function = modulation_function

    @property
    def modulation_function(self):
        return self._modulation_function


@pytest.fixture
def kernel():
    rw_mats = [
        torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        ),
        torch.tensor(
            [
                [0.0, 2.0],
                [3.0, 0.0],
            ]
        ),
    ]
    modulation_function = torch.tensor([2.0, -1.0])
    return DummyGRFKernel(rw_mats=rw_mats, modulation_function=modulation_function)


def test_base_grf_kernel_builds_feature_matrix(kernel):
    expected = torch.tensor(
        [
            [2.0, -2.0],
            [-3.0, 2.0],
        ]
    )

    assert torch.equal(kernel._get_feature_matrix(), expected)


@pytest.mark.parametrize(
    ("x1_idx", "x2_idx", "diag", "expected"),
    [
        (
            None,
            None,
            False,
            torch.tensor(
                [
                    [8.0, -10.0],
                    [-10.0, 13.0],
                ]
            ),
        ),
        (
            torch.tensor([1, 0]),
            torch.tensor([0]),
            False,
            torch.tensor(
                [
                    [-10.0],
                    [8.0],
                ]
            ),
        ),
        (
            torch.tensor([0, 1]),
            torch.tensor([1, 0]),
            True,
            torch.tensor([-10.0, -10.0]),
        ),
    ],
)
def test_base_grf_kernel_forward(kernel, x1_idx, x2_idx, diag, expected):
    result = kernel.forward(x1_idx=x1_idx, x2_idx=x2_idx, diag=diag)

    assert torch.equal(result, expected)
