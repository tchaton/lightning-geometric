import os.path as osp
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from examples.core.base_model import BaseModel

class GCNConvLinkPredNet(BaseModel):
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
                kwargs["embedding_dim"],
                cached=kwargs["cached"],
                normalize=not kwargs["use_gdc"],
            )
        )

    def forward(self, batch):
        x = batch.x.squeeze()
        for idx, conv in enumerate(self.convs):
            x = F.relu(conv(x, batch.pos_edge_index.squeeze(), batch.pos_edge_attr))
        total_edge_index = torch.cat([batch.pos_edge_index.squeeze(), batch.neg_edge_index.squeeze()], dim=-1)
        x_j = torch.index_select(x, 0, total_edge_index[0])
        x_i = torch.index_select(x, 0, total_edge_index[1])
        return torch.einsum("ef, ef -> e", x_i, x_j), 0