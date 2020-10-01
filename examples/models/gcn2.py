import os.path as osp
from functools import partial
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCN2Conv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import pytorch_lightning as pl
from examples.core.base_model import BaseModel


class GCN2ConvNet(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.save_hyperparameters()

        assert kwargs["num_layers"] >= 2

        self.convs = nn.ModuleList()
        self.lins = torch.nn.ModuleList()

        self.make_layers = partial(
            self.make_layers,
            alpha=kwargs["alpha"],
            theta=kwargs["theta"],
            shared_weights=kwargs["shared_weights"],
            normalize=kwargs["normalize"],
        )

        self.make_layers(kwargs["num_features"], kwargs["hidden_channels"])

        for idx in range(kwargs["num_layers"] - 2):
            self.make_layers(kwargs["hidden_channels"], kwargs["hidden_channels"])

        self.lins.append(
            torch.nn.Linear(kwargs["hidden_channels"], kwargs["num_classes"])
        )

    def make_layers(
        self,
        dim_in,
        dim_out,
        alpha=None,
        theta=None,
        shared_weights=None,
        normalize=None,
    ):
        self.convs.append(
            GCN2Conv(dim_out, alpha, theta, shared_weights, normalize=normalize)
        )
        self.lins.append(torch.nn.Linear(dim_in, dim_out))

    def forward(self, batch):
        x = batch.x
        x_0 = batch.x
        for idx, (conv, lin) in enumerate(zip(self.convs, self.lins)):
            x = lin(x)
            if idx == 0:
                x_0 = x
            edge_index = (
                batch.edge_index[idx]
                if len(batch.edge_index) == 1
                else batch.edge_index[0]
            )
            x = conv(x, x_0, edge_index, batch.edge_attr)
        return lin(x), 0