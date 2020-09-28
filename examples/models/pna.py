import os.path as osp

import torch
from torch import nn
from torch.nn import ModuleList, Embedding
from torch.nn import Sequential, ReLU, Linear
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from torch_geometric.nn import PNAConv, BatchNorm, global_add_pool
import pytorch_lightning as pl
from examples.core.base_model import BaseModel


class PNAConvNet(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.save_hyperparameters()

        self.node_emb = Embedding(kwargs["node_vocab"], kwargs["node_dim"])
        self.edge_emb = Embedding(kwargs["edge_vocab"], kwargs["edge_dim"])

        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        for _ in range(kwargs["num_layers"]):
            conv = PNAConv(
                in_channels=kwargs["node_dim"],
                out_channels=kwargs["node_dim"],
                aggregators=kwargs["aggregators"],
                scalers=kwargs["scalers"],
                deg=torch.tensor(kwargs["deg"]),
                edge_dim=kwargs["edge_dim"],
                towers=kwargs["towers"],
                pre_layers=kwargs["pre_layers"],
                post_layers=kwargs["post_layers"],
                divide_input=kwargs["divide_input"],
            )
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(kwargs["node_dim"]))

        self.mlp = Sequential(
            Linear(kwargs["node_dim"], kwargs["edge_dim"]),
            ReLU(),
            Linear(kwargs["edge_dim"], kwargs["hidden_channels"]),
            ReLU(),
            Linear(kwargs["hidden_channels"], kwargs["num_classes"]),
        )

    def forward(self, batch):
        batch_idx = batch.batch
        x = batch.x
        edge_attr = batch.edge_attr

        x = self.node_emb(x.squeeze())
        edge_attr = self.edge_emb(edge_attr)

        for idx, (conv, batch_norm) in enumerate(zip(self.convs, self.batch_norms)):
            edge_index = (
                batch.edge_index[idx]
                if isinstance(batch.edge_index, list)
                else batch.edge_index
            )
            x = F.relu(batch_norm(conv(x, edge_index, edge_attr)))

        x = global_add_pool(x, batch_idx)

        return self.mlp(x), 0