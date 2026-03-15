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
    """Run random walks for a worker process.

    :param args: Packed worker arguments containing node chunks and walk settings.
    :returns: Per-step accumulators keyed by ``(start_node, current_node)`` pairs.
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
    """Execute the core random-walk loop for a set of start nodes.

    :param nodes: Start nodes assigned to the current worker.
    :param walks_per_node: Number of walks sampled from each start node.
    :param p_halt: Probability of terminating a walk after each step.
    :param max_walk_length: Maximum number of recorded walk steps.
    :param seed: Base random seed for deterministic per-node sampling.
    :param show_progress: Whether to display a progress bar for this worker.
    :returns: Per-step accumulators keyed by ``(start_node, current_node)`` pairs.
    :raises RuntimeError: If the worker CSR arrays have not been initialized.
    """
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
    """Bind CSR arrays in a worker process.

    :param crow: CSR row pointer array.
    :param col: CSR column index array.
    :param data: CSR nonzero value array.
    :returns: ``None``.
    """
    global _G_CROW, _G_COL, _G_DATA
    _G_CROW = crow
    _G_COL = col
    _G_DATA = data


class GRFSampler:
    """Generate GRF random-walk matrices as sparse linear operators."""

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
        """Initialize the GRF random-walk sampler.

        :param adjacency_matrix: Square adjacency matrix of the graph.
        :param walks_per_node: Number of walks sampled from each node.
        :param p_halt: Probability of stopping after each walk step.
        :param max_walk_length: Maximum number of steps retained per walk.
        :param seed: Random seed used for deterministic sampling.
        :param use_tqdm: Whether to display progress bars during sampling.
        :param n_processes: Number of worker processes to use.
        :returns: ``None``.
        :raises ValueError: If ``adjacency_matrix`` is not square.
        """
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
        """Sample per-step random-walk matrices.

        :returns: A list of per-step random-walk matrices wrapped as
            :class:`~grf_gp.utils.sparse_lo.SparseLinearOperator` objects.
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
        """Sample per-step random-walk matrices.

        :returns: A list of per-step random-walk matrices.
        """
        return self.sample_random_walk_matrices()
