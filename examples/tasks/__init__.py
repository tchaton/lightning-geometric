from typing import *
import torch
from torch_sparse import SparseTensor


class Batch(NamedTuple):
    x: torch.Tensor
    edge_index: List[SparseTensor]