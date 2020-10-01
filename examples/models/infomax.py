import os.path as osp
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, DeepGraphInfomax
from examples.core.base_model import BaseModel


class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(Encoder, self).__init__()
        self.conv = GCNConv(in_channels, hidden_channels, cached=True)
        self.prelu = nn.PReLU(hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.prelu(x)
        return x


def corruption(x, edge_index):
    return x[torch.randperm(x.size(0))], edge_index


class InfoMaxModel(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.save_hyperparameters()

        self.model = DeepGraphInfomax(
            hidden_channels=kwargs["hidden_channels"],
            encoder=Encoder(kwargs["num_features"], kwargs["hidden_channels"]),
            summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
            corruption=corruption,
        )

    def forward(self, batch):
        pos_z, neg_z, summary = self.model(batch.x, batch.edge_index[0])
        loss = self.model.loss(pos_z, neg_z, summary)
        return pos_z, loss