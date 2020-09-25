import os.path as osp

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import DNAConv
import pytorch_lightning as pl
from examples.core.base_model import BaseModel


class DNAConvNet(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.save_hyperparameters()

        self.hidden_channels = kwargs["hidden_channels"]
        self.lin1 = torch.nn.Linear(kwargs["num_features"], kwargs["hidden_channels"])
        self.convs = torch.nn.ModuleList()
        for _ in range(kwargs["num_layers"]):
            self.convs.append(
                DNAConv(
                    kwargs["hidden_channels"],
                    kwargs["heads"],
                    kwargs["groups"],
                    dropout=kwargs["dropout"],
                    cached=False,
                )
            )
        self.lin2 = torch.nn.Linear(kwargs["hidden_channels"], kwargs["num_classes"])

    def forward(self, x, adjs):
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x_all = x.view(-1, 1, self.hidden_channels)
        for idx, conv in enumerate(self.convs):
            print(x_all.shape, adjs[idx].edge_index.max())
            x = F.relu(conv(x_all, adjs[idx].edge_index))
            x = x.view(-1, 1, self.hidden_channels)
            x_all = torch.cat([x_all, x], dim=1)
        x = x_all[:, -1]
        x = F.dropout(x, p=0.5, training=self.training)
        return self.lin2(x)
