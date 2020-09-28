from typing import *
from torch import nn
from typing import *
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from examples.core.base_model import BaseModel


class SAGEConvNet(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.save_hyperparameters()

        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(kwargs["num_features"], kwargs["hidden_channels"]))
        for _ in range(kwargs["num_layers"] - 2):
            self.convs.append(
                SAGEConv(kwargs["hidden_channels"], kwargs["hidden_channels"])
            )
        self.convs.append(SAGEConv(kwargs["hidden_channels"], kwargs["num_classes"]))

    def forward(self, batch):
        x = batch.x
        for idx, conv in enumerate(self.convs):
            edge_index = (
                batch.edge_index[0]
                if len(batch.edge_index[idx]) == 1
                else batch.edge_index[idx]
            )
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, training=self.training)
        return x, 0