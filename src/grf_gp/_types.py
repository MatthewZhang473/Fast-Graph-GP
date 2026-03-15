from __future__ import annotations

from typing import Sequence, TypeAlias

import torch
from jaxtyping import Float, Int
from linear_operator.operators import LinearOperator

Tensor: TypeAlias = torch.Tensor
Device: TypeAlias = torch.device

NodeIndexTensor: TypeAlias = Int[Tensor, "n"]
ObservationTensor: TypeAlias = Float[Tensor, "n"]
SampleTensor: TypeAlias = Float[Tensor, "batch n"]
DenseMatrix: TypeAlias = Float[Tensor, "rows cols"]
DenseVector: TypeAlias = Float[Tensor, "n"]
ScalarTensor: TypeAlias = Float[Tensor, ""]

FeatureMatrixLike: TypeAlias = Tensor | LinearOperator
RandomWalkMatrixLike: TypeAlias = Tensor | LinearOperator
RandomWalkMatrices: TypeAlias = Sequence[RandomWalkMatrixLike]
