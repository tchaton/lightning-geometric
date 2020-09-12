import os.path as osp

import torch
from torch import nn
from torch.nn import ModuleList, Embedding
from torch.nn import Sequential, ReLU, Linear
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from torch_geometric.nn import PNAConv, BatchNorm, global_add_pool
import pytorch_lightning as pl


class PNAConvNet(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.save_hyperparameters()

        self.node_emb = Embedding(kwargs["node_vocab"], kwargs["node_dim"])
        self.edge_emb = Embedding(kwargs["edge_vocab"], kwargs["edge_dim"])

        aggregators = ["mean", "min", "max", "std"]
        scalers = ["identity", "amplification", "attenuation"]

        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        for _ in range(4):
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
            Linear(kwargs["edge_dim"], 25),
            ReLU(),
            Linear(25, kwargs["num_classes"]),
        )

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.node_emb(x.squeeze())
        edge_attr = self.edge_emb(edge_attr)

        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index, edge_attr)))

        x = global_add_pool(x, batch)

        return self.mlp(x)

    def configure_optimizers(self):
        return self._init_optim(self.parameters())