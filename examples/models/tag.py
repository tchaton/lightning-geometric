import os.path as osp

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import TAGConv
import pytorch_lightning as pl
from examples.models.base_model import BaseModel


class TAGConvNet(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.save_hyperparameters()

        self.conv1 = TAGConv(kwargs["num_features"], kwargs["hidden_channels"])
        self.conv2 = TAGConv(kwargs["hidden_channels"], kwargs["num_classes"])

    def forward(self, x, adjs):
        x = F.relu(self.conv1(x, adjs[0].edge_index))
        x = F.dropout(x, training=self.training)
        return self.conv2(x, adjs[1].edge_index)

    def configure_optimizers(self):
        return self._init_optim(self.parameters())