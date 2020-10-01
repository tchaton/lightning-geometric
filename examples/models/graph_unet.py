import os.path as osp
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GraphUNet
from examples.core.base_model import BaseModel


class GraphUNetModel(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.save_hyperparameters()

        assert kwargs["num_layers"] >= 2

        self.unet = GraphUNet(
            kwargs["num_features"],
            kwargs["hidden_channels"],
            kwargs["num_classes"],
            depth=kwargs["num_layers"],
            pool_ratios=kwargs["pool_ratios"],
        )

    def forward(self, batch):
        x = self.unet(batch.x, batch.edge_index[0])
        return x, 0