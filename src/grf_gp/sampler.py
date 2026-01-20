"""GRF random-walk sampler built around torch sparse CSR tensors."""

import os
import multiprocessing as mp
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional, Union

import numpy as np
import torch
from tqdm.auto import tqdm

from .utils.sparse_lo import SparseLinearOperator
from .utils.csr import to_sparse_csr, build_csr_from_entries


def _worker_walks(
    args: tuple,
) -> List[defaultdict]:
    """
    Worker function for multiprocessing random walks.
    """
    (
        nodes,
        walks_per_node,
        p_halt,
        max_walk_length,
        seed,
        show_progress,
    ) = args
    return _run_walks(
        nodes=np.asarray(nodes),
        walks_per_node=walks_per_node,
        p_halt=p_halt,
        max_walk_length=max_walk_length,
        seed=seed,
        show_progress=show_progress,
    )


# Globals for worker fast access
_G_CROW: Optional[np.ndarray] = None
_G_COL: Optional[np.ndarray] = None
_G_DATA: Optional[np.ndarray] = None


def _run_walks(
    nodes: np.ndarray,
    walks_per_node: int,
    p_halt: float,
    max_walk_length: int,
    seed: int,
    show_progress: bool,
) -> List[defaultdict]:
    """Core random-walk loop shared by worker processes."""
    if _G_CROW is None or _G_COL is None or _G_DATA is None:
        raise RuntimeError("CSR arrays are not available in this process.")
    crow, col, data = _G_CROW, _G_COL, _G_DATA
    step_accumulators: List[defaultdict] = [
        defaultdict(float) for _ in range(max_walk_length)
    ]

    iterator = tqdm(nodes, desc="Process walks", disable=not show_progress)
    for start_node in iterator:
        start_node = int(start_node)
        rng = np.random.default_rng(seed + start_node)  # per-node seed for determinism
        for _ in range(walks_per_node):
            current_node = start_node
            load = 1.0
            for step in range(max_walk_length):
                step_accumulators[step][(start_node, current_node)] += load

                start = crow[current_node]
                end = crow[current_node + 1]
                degree = end - start
                if degree == 0:
                    break
                if rng.random() < p_halt:
                    break
                offset = rng.integers(degree)
                weight = data[start + offset]
                current_node = int(col[start + offset])
                load *= degree * weight / (1 - p_halt)

    return step_accumulators


def _init_worker(crow: np.ndarray, col: np.ndarray, data: np.ndarray) -> None:
    """Initializer for worker processes (bind CSR arrays)."""
    global _G_CROW, _G_COL, _G_DATA
    _G_CROW = crow
    _G_COL = col
    _G_DATA = data


class GRFSampler:
    """
    Generates GRF random walk matrices
    and returns them as SparseLinearOperator objects.
    """

    def __init__(
        self,
        adjacency_matrix: Union[torch.Tensor, "torch.sparse.Tensor"],
        walks_per_node: int = 10,
        p_halt: float = 0.5,
        max_walk_length: int = 10,
        seed: Optional[int] = None,
        use_tqdm: bool = True,
        n_processes: Optional[int] = None,
    ) -> None:
        self.adjacency_csr = to_sparse_csr(adjacency_matrix).cpu()
        if self.adjacency_csr.size(0) != self.adjacency_csr.size(1):
            raise ValueError("Adjacency matrix must be square.")

        self.walks_per_node = walks_per_node
        self.p_halt = p_halt
        self.max_walk_length = max_walk_length
        self.use_tqdm = use_tqdm
        self.n_processes = n_processes
        self.seed = seed or 42

    def sample_random_walk_matrices(self) -> List[SparseLinearOperator]:
        """
        Perform GRF random walks and return per-step random walk matrices.
        """
        crow_indices = self.adjacency_csr.crow_indices().numpy()
        col_indices = self.adjacency_csr.col_indices().numpy()
        values = self.adjacency_csr.values().numpy()
        num_nodes = self.adjacency_csr.size(0)

        n_proc = self.n_processes or os.cpu_count() or 1
        chunks = np.array_split(np.arange(num_nodes), n_proc)

        ctx = mp.get_context("fork")

        with ProcessPoolExecutor(
            max_workers=n_proc,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(crow_indices, col_indices, values),
        ) as executor:
            args = [
                (
                    chunk.tolist(),
                    self.walks_per_node,
                    self.p_halt,
                    self.max_walk_length,
                    self.seed + i,
                    self.use_tqdm and i == 0,
                )
                for i, chunk in enumerate(chunks)
            ]
            futures = [executor.submit(_worker_walks, a) for a in args]
            results = [fut.result() for fut in as_completed(futures)]

        accumulators = [defaultdict(float) for _ in range(self.max_walk_length)]
        for result in results:
            for step in range(self.max_walk_length):
                for key, val in result[step].items():
                    accumulators[step][key] += val

        matrices = [
            SparseLinearOperator(
                build_csr_from_entries(num_nodes, acc) * (1.0 / self.walks_per_node)
            )
            for acc in accumulators
        ]
        return matrices

    def __call__(self) -> List[SparseLinearOperator]:
        return self.sample_random_walk_matrices()
