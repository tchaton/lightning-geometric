import os.path as osp

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import pytorch_lightning as pl
from examples.core.base_model import BaseModel


class GATConvNet(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.save_hyperparameters()

        assert kwargs["num_layers"] >= 2

        self._num_layers = kwargs["num_layers"]

        self.convs = nn.ModuleList()
        self.lins = nn.ModuleList()

        num_features = kwargs["num_features"]
        hidden_channels = kwargs["hidden_channels"]
        num_classes = kwargs["num_classes"]
        heads = kwargs["heads"]

        self.make_layers(num_features, hidden_channels, heads=heads)

        for idx in range(self._num_layers - 2):
            self.make_layers(
                hidden_channels * heads,
                hidden_channels,
                heads=heads,
            )

        self.make_layers(hidden_channels * heads, num_classes, heads=1)

        self.elu = nn.ELU()
        self.identity = nn.Identity()
        self.acts = nn.ModuleList(
            [self.elu for _ in range(self._num_layers - 1)] + [self.identity]
        )

    def make_layers(self, dim_in, dim_out, heads=4):
        self.convs.append(
            GATConv(
                dim_in,
                dim_out,
                heads=heads,
            )
        )
        self.lins.append(torch.nn.Linear(dim_in, heads * dim_out))

    def forward(self, batch):
        x = batch.x
        for idx, (conv, lin, act) in enumerate(zip(self.convs, self.lins, self.acts)):
            x = act(conv(x, batch.edge_index[idx]) + lin(x))
        return x, 0