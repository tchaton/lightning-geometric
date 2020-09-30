import os.path as osp

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import ARMAConv
import pytorch_lightning as pl
from examples.core.base_model import BaseModel


class ARMAConvNet(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.save_hyperparameters()

        self.convs = nn.ModuleList()

        self.convs.append(
            ARMAConv(
                kwargs["num_features"],
                kwargs["hidden_channels"],
                num_stacks=kwargs["num_stacks"],
                num_layers=kwargs["arma_num_layers"],
                shared_weights=kwargs["shared_weights"],
                dropout=kwargs["dropout"],
                act=nn.LeakyReLU(),
            )
        )

        for idx in range(kwargs["num_layers"] - 2):
            self.convs.append(
                ARMAConv(
                    kwargs["hidden_channels"],
                    kwargs["hidden_channels"],
                    num_stacks=kwargs["num_stacks"],
                    num_layers=kwargs["arma_num_layers"],
                    shared_weights=kwargs["shared_weights"],
                    dropout=kwargs["dropout"],
                    act=nn.LeakyReLU(),
                )
            )

        self.convs.append(
            ARMAConv(
                kwargs["hidden_channels"],
                kwargs["num_classes"],
                num_stacks=kwargs["num_stacks"],
                num_layers=kwargs["arma_num_layers"],
                shared_weights=kwargs["shared_weights"],
                dropout=kwargs["dropout"],
                act=nn.Identity(),
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
            x = conv(x, edge_index)
        return x, 0