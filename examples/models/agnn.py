import os.path as osp

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import AGNNConv
import pytorch_lightning as pl


class AGNNConvNet(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.save_hyperparameters()

        self.lin1 = torch.nn.Linear(kwargs["num_features"], 16)
        self.prop1 = AGNNConv(requires_grad=False)
        self.prop2 = AGNNConv(requires_grad=True)
        self.lin2 = torch.nn.Linear(16, kwargs["num_classes"])

    def forward(self, x, adjs):
        x = F.dropout(x, training=self.training)
        x = F.relu(self.lin1(x))
        x = self.prop1(x, adjs[0].edge_index)
        x = self.prop2(x, adjs[1].edge_index)
        x = F.dropout(x, training=self.training)
        x = self.lin2(x)
        return x

    def configure_optimizers(self):
        return self._init_optim(self.parameters())