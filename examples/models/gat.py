import os.path as osp

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import pytorch_lightning as pl
from sklearn.metrics import f1_score


class GATConvNet(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.save_hyperparameters()

        self.conv1 = GATConv(kwargs["num_features"], 256, heads=4)
        self.lin1 = torch.nn.Linear(kwargs["num_features"], 4 * 256)
        self.conv2 = GATConv(4 * 256, 256, heads=4)
        self.lin2 = torch.nn.Linear(4 * 256, 4 * 256)
        self.conv3 = GATConv(4 * 256, kwargs["num_classes"], heads=6, concat=False)
        self.lin3 = torch.nn.Linear(4 * 256, kwargs["num_classes"])

        self.loss_op = torch.nn.BCEWithLogitsLoss()

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index) + self.lin1(x))
        x = F.elu(self.conv2(x, edge_index) + self.lin2(x))
        x = self.conv3(x, edge_index) + self.lin3(x)
        return x

    def training_step(self, batch, batch_nb):
        loss = batch.num_graphs * self.loss_op(
            self.forward(batch.x, batch.edge_index), batch.y
        )
        result = pl.TrainResult(loss)
        result.log("train_loss", loss, prog_bar=True)
        return result

    def validation_step(self, batch, batch_nb):
        preds = self.forward(batch.x, batch.edge_index)
        loss = batch.num_graphs * self.loss_op(preds, batch.y)
        result = pl.EvalResult(loss)
        result.log("val_loss", loss)
        result.y = batch.y
        result.preds = preds
        result.num_graphs = torch.tensor([batch.num_graphs]).float()
        return result

    def validation_epoch_end(self, outputs):
        avg_loss = outputs["val_loss"].sum() / outputs["num_graphs"].sum()
        val_f1_score = torch.tensor(
            [f1_score(outputs["y"], outputs["preds"] > 0, average="micro")]
        )
        result = pl.EvalResult(checkpoint_on=val_f1_score)
        result.log("val_loss", avg_loss, prog_bar=True)
        result.log("val_f1_score", val_f1_score, prog_bar=True)
        return result

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)