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

        self.forward = (
            self.forward_with_gdc if kwargs["use_gdc"] else self.forward_without_gcd
        )

    def forward_with_gdc(self, batch):
        x = batch.x
        edge_attr = batch.edge_attr
        assert edge_attr is not None
        edge_attr = edge_attr[0]
        for idx, conv in enumerate(self.convs):
            edge_index = (
                batch.edge_index[idx]
                if len(batch.edge_index) == 1
                else batch.edge_index[0]
            )
            x = conv(x, edge_index, edge_attr)
        return x, 0

    def forward_without_gcd(self, batch):
        x = batch.x
        assert batch.edge_attr is None
        edge_attr = None
        for idx, conv in enumerate(self.convs):
            edge_index = (
                batch.edge_index[idx]
                if len(batch.edge_index) == 1
                else batch.edge_index[0]
            )
            x = conv(x, edge_index, edge_attr)
        return x, 0