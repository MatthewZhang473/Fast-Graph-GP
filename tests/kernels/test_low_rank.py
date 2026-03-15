import torch

from grf_gp.kernels.low_rank import LowRankGRFKernel


class DummyLowRankGRFKernel(LowRankGRFKernel):
    def __init__(self, rw_mats, modulation_function, proj_dim, jlt_seed=42):
        super().__init__(
            rw_mats=rw_mats,
            proj_dim=proj_dim,
            jlt_seed=jlt_seed,
        )
        self._modulation_function = modulation_function

    @property
    def modulation_function(self):
        return self._modulation_function


def test_low_rank_grf_kernel_projects_feature_matrix():
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
    kernel = DummyLowRankGRFKernel(
        rw_mats=rw_mats,
        modulation_function=modulation_function,
        proj_dim=1,
        jlt_seed=7,
    )

    phi_full = modulation_function[0] * rw_mats[0] + modulation_function[1] * rw_mats[1]
    expected = phi_full @ kernel.jlt_proj

    assert torch.allclose(kernel._get_feature_matrix(), expected)


def test_low_rank_grf_kernel_jlt_seed_is_deterministic():
    rw_mats = [torch.eye(2)]
    modulation_function = torch.tensor([1.0])

    kernel_a = DummyLowRankGRFKernel(
        rw_mats=rw_mats,
        modulation_function=modulation_function,
        proj_dim=2,
        jlt_seed=7,
    )
    kernel_b = DummyLowRankGRFKernel(
        rw_mats=rw_mats,
        modulation_function=modulation_function,
        proj_dim=2,
        jlt_seed=7,
    )

    assert torch.equal(kernel_a.jlt_proj, kernel_b.jlt_proj)
