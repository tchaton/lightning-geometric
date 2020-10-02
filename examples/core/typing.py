from typing import *
from torch import Tensor
from torch_sparse import SparseTensor

class EdgeBatch(NamedTuple):
    x: Tensor
    pos_edge_index: Tensor
    neg_edge_index: Tensor
    edge_attr: Optional[Tensor]
    batch: Optional[Tensor]

class SparseBatch(NamedTuple):
    x: Tensor
    edge_index: List[SparseTensor]
    edge_attr: Optional[Tensor]
    batch: Optional[Tensor]


class TensorBatch(NamedTuple):
    x: Tensor
    edge_index: List[Tensor]
    edge_attr: Optional[Tensor]
    batch: Optional[Tensor]