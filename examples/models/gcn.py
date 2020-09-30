import os.path as osp

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import pytorch_lightning as pl
from examples.core.base_model import BaseModel
from examples.core.base_model import BaseModel


class GCNConvNet(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.save_hyperparameters()

        assert kwargs["num_layers"] >= 2

        self.convs = nn.ModuleList()

        self.convs.append(
            GCNConv(
                kwargs["num_features"],
                kwargs["hidden_channels"],
                cached=kwargs["cached"],
                normalize=not kwargs["use_gdc"],
            )
        )

        for idx in range(kwargs["num_layers"] - 2):
            self.convs.append(
                GCNConv(
                    kwargs["hidden_channels"],
                    kwargs["hidden_channels"],
                    cached=kwargs["cached"],
                    normalize=not kwargs["use_gdc"],
                )
            )

        self.convs.append(
            GCNConv(
                kwargs["hidden_channels"],
                kwargs["num_classes"],
                cached=kwargs["cached"],
                normalize=not kwargs["use_gdc"],
            )
        )

    def forward(self, batch):
        x = batch.x
        for idx, conv in enumerate(self.convs):
            edge_index = (
                batch.edge_index[idx]
                if len(batch.edge_index) == 1
                else batch.edge_index[0]
            )
            if batch.edge_attr is not None:
                edge_attr = (
                    batch.edge_attr[idx]
                    if len(batch.edge_attr) == 1
                    else batch.edge_attr[0]
                )
            else:
                edge_attr = None
            x = conv(x, edge_index, edge_attr)
        return x, 0