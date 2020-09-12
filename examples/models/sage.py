import os.path as osp

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import pytorch_lightning as pl


class SAGEConvNet(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.save_hyperparameters()
        self.num_layers = 2
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(kwargs["num_features"], kwargs["hidden_channels"]))
        self.convs.append(
            SAGEConv(kwargs["hidden_channels"], kwargs["hidden_channels"])
        )
        self.lin = torch.nn.Linear(kwargs["hidden_channels"], kwargs["num_classes"])

    def forward(self, x, adjs):
        x = F.dropout(x, training=self.training)
        x = self.convs[0](x, adjs[0].edge_index)
        x = self.convs[1](x, adjs[1].edge_index)
        x = F.dropout(x, training=self.training)
        return self.lin(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)