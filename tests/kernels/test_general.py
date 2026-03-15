import torch

from grf_gp.kernels.general import GeneralGRFKernel


def test_general_grf_kernel_forward():
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
    kernel = GeneralGRFKernel(rw_mats=rw_mats, max_walk_length=2)

    with torch.no_grad():
        kernel.raw_modulation_function.copy_(torch.tensor([2.0, -1.0]))

    expected = torch.tensor(
        [
            [8.0, -10.0],
            [-10.0, 13.0],
        ]
    )

    assert torch.equal(kernel.forward(), expected)
