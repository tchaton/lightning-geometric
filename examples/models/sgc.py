import os.path as osp

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import SGConv
import pytorch_lightning as pl
from examples.models.base_model import BaseModel


class SGConvNet(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.save_hyperparameters()

        self.conv = SGConv(
            kwargs["num_features"],
            kwargs["num_classes"],
            K=kwargs["K"],
            cached=kwargs["cached"],
        )

    def forward(self, x, adjs):
        return self.conv(x, adjs[0].edge_index)

    def configure_optimizers(self):
        return self._init_optim(self.parameters())