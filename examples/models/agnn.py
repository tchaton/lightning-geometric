import os.path as osp

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import AGNNConv
import pytorch_lightning as pl

class AGNNConvNet(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.save_hyperparameters()

        self.lin1 = torch.nn.Linear(kwargs["num_features"], 16)
        self.prop1 = AGNNConv(requires_grad=False)
        self.prop2 = AGNNConv(requires_grad=True)
        self.lin2 = torch.nn.Linear(16, kwargs["num_classes"])

    def forward(self):
        x = F.dropout(data.x, training=self.training)
        x = F.relu(self.lin1(x))
        x = self.prop1(x, data.edge_index)
        x = self.prop2(x, data.edge_index)
        x = F.dropout(x, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_nb):
        if isinstance(batch, (list, tuple)):
            x, y = batch
        else:
            x = batch[:, 1:]
            y = batch[:, 0].long()
        loss = F.nll_loss(F.log_softmax(self.forward(x), -1), y)
        result = pl.TrainResult(loss)
        result.log('train_loss', loss)
        return result

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)