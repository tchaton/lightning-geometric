import os.path as osp
import torch
from torch import nn
from typing import *
from collections import namedtuple
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.nn import SAGEConv
import pytorch_lightning as pl
from examples.core.base_model import BaseModel


class SAGEConvNet(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.save_hyperparameters()

        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(kwargs["num_features"], kwargs["hidden_channels"]))
        for _ in range(kwargs["num_layers"] - 2):
            self.convs.append(
                SAGEConv(kwargs["hidden_channels"], kwargs["hidden_channels"])
            )
        self.convs.append(SAGEConv(kwargs["hidden_channels"], kwargs["num_classes"]))

    def forward(self, batch):
        x = batch.x
        for idx, conv in enumerate(self.convs):
            x = F.relu(conv(x, batch.edge_index[idx]))
            x = F.dropout(x, training=self.training)
        return x, 0