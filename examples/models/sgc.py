import os.path as osp

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import SGConv
import pytorch_lightning as pl
from examples.core.base_model import BaseModel


class SGConvNet(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.save_hyperparameters()

        self.convs = nn.ModuleList()
        self.convs.append(SGConv(kwargs["num_features"], kwargs["hidden_channels"]))
        for _ in range(kwargs["num_layers"] - 2):
            self.convs.append(
                SGConv(
                    kwargs["hidden_channels"],
                    kwargs["hidden_channels"],
                    K=kwargs["K"],
                    cached=kwargs["cached"],
                )
            )
        self.convs.append(SGConv(kwargs["hidden_channels"], kwargs["num_classes"]))

    def forward(self, x, adjs):
        for idx, conv in enumerate(self.convs):
            x = F.relu(conv(x, adjs[idx].edge_index))
            x = F.dropout(x, training=self.training)
        return x