from collections import defaultdict
from typing import Union
import torch

def to_sparse_csr(
    adjacency: Union[torch.Tensor, "torch.sparse.Tensor"],
) -> torch.Tensor:
    """
    Coerce input adjacency to a torch.sparse_csr_tensor.
    """
    if isinstance(adjacency, torch.Tensor) and adjacency.is_sparse_csr:
        return adjacency
    if isinstance(adjacency, torch.Tensor) and adjacency.is_sparse:
        return adjacency.to_sparse_csr()
    if isinstance(adjacency, torch.Tensor):
        return adjacency.to_sparse_csr()
    raise TypeError("adjacency must be a torch Tensor (dense or sparse)")

def build_csr_from_entries(num_nodes: int, entries: defaultdict) -> torch.Tensor:
    if not entries:
        crow = torch.zeros(num_nodes + 1, dtype=torch.int64)
        col = torch.zeros(0, dtype=torch.int64)
        vals = torch.zeros(0, dtype=torch.float32)
        return torch.sparse_csr_tensor(crow, col, vals, (num_nodes, num_nodes))

    keys = list(entries.keys())
    rows = torch.tensor([k[0] for k in keys], dtype=torch.int64)
    cols = torch.tensor([k[1] for k in keys], dtype=torch.int64)
    vals = torch.tensor([entries[k] for k in keys], dtype=torch.float32)
    # Torch expects crow_indices to be monotonic; use coo -> csr for simplicity
    coo = torch.sparse_coo_tensor(
        torch.stack([rows, cols]), vals, size=(num_nodes, num_nodes)
    ).coalesce()
    return coo.to_sparse_csr()
