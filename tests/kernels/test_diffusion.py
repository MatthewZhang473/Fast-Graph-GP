import torch

from grf_gp.kernels.diffusion import DiffusionGRFKernel, diffusion_formula


def test_diffusion_grf_kernel_modulation_function():
    rw_mats = [
        torch.eye(2),
        torch.tensor(
            [
                [0.0, 1.0],
                [1.0, 0.0],
            ]
        ),
    ]
    kernel = DiffusionGRFKernel(rw_mats=rw_mats, max_walk_length=2)

    with torch.no_grad():
        kernel.raw_beta.fill_(0.0)
        kernel.raw_sigma_f.fill_(0.0)

    expected = kernel.sigma_f * diffusion_formula(
        torch.arange(2, dtype=kernel.raw_beta.dtype),
        kernel.beta,
    )

    assert torch.allclose(kernel.modulation_function, expected)


def test_diffusion_grf_kernel_forward():
    rw_mats = [
        torch.eye(2),
        torch.tensor(
            [
                [0.0, 1.0],
                [1.0, 0.0],
            ]
        ),
    ]
    kernel = DiffusionGRFKernel(rw_mats=rw_mats, max_walk_length=2)

    with torch.no_grad():
        kernel.raw_beta.fill_(0.0)
        kernel.raw_sigma_f.fill_(0.0)

    modulation = kernel.modulation_function
    phi = modulation[0] * rw_mats[0] + modulation[1] * rw_mats[1]
    expected = phi @ phi.transpose(-1, -2)

    assert torch.allclose(kernel.forward(), expected)
