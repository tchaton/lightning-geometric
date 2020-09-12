import os.path as osp
from hydra.utils import instantiate
import torch
from torch import nn
from torch.nn import ModuleList, Embedding
from torch.nn import Sequential, ReLU, Linear
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from torch_geometric.nn import PNAConv, BatchNorm, global_add_pool
import pytorch_lightning as pl
from examples.models.base_model import BaseModel


class RENet(BaseModel):
    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)

        model.params.num_nodes = kwargs.get("num_nodes")
        model.params.num_rels = kwargs.get("num_rels")
        model.params.seq_len = model.params.seq_len

        self.save_hyperparameters()
        self.model = instantiate(model)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return self._init_optim(self.parameters())