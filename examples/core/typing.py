from typing import *
from torch import Tensor
from torch_sparse import SparseTensor


class SparseBatch(NamedTuple):
    x: Tensor
    edge_index: List[SparseTensor]


class TensorBatch(NamedTuple):
    x: Tensor
    edge_index: List[Tensor]