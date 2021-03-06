import os.path as osp

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import AGNNConv
import pytorch_lightning as pl
from examples.core.base_model import BaseModel


class AGNNConvNet(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.save_hyperparameters()

        self.mlp_in = torch.nn.Linear(kwargs["num_features"], 16)
        self.convs = nn.ModuleList()
        for _ in range(kwargs["num_layers"]):
            self.convs.append(AGNNConv(requires_grad=True))
        self.mlp_out = torch.nn.Linear(16, kwargs["num_classes"])

    def forward(self, batch):
        x = batch.x
        x = F.dropout(x, training=self.training)
        x = F.relu(self.mlp_in(x))
        for idx, conv in enumerate(self.convs):
            x = conv(x, batch.edge_index[idx])
        x = F.dropout(x, training=self.training)
        return self.mlp_out(x), 0