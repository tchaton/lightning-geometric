import os.path as osp

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import pytorch_lightning as pl
from examples.core.base_model import BaseModel


class GATConvNet(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.save_hyperparameters()

        self.conv1 = GATConv(kwargs["num_features"], 256, heads=4)
        self.lin1 = torch.nn.Linear(kwargs["num_features"], 4 * 256)
        self.conv2 = GATConv(4 * 256, 256, heads=4)
        self.lin2 = torch.nn.Linear(4 * 256, 4 * 256)
        self.conv3 = GATConv(4 * 256, kwargs["num_classes"], heads=6, concat=False)
        self.lin3 = torch.nn.Linear(4 * 256, kwargs["num_classes"])

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index) + self.lin1(x))
        x = F.elu(self.conv2(x, edge_index) + self.lin2(x))
        return self.conv3(x, edge_index) + self.lin3(x)