import os
import os.path as osp
import numpy as np
from functools import partial
import torch
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import LightningDataModule
import torch_geometric
from torch_geometric.datasets import ZINC
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import f1_score
from torch_geometric.utils import degree
from examples.core.base_dataset import BaseDataset


class ZINCDataset(BaseDataset):

    NAME = "ZINC"

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )

    @property
    def node_dim(self):
        return 75

    @property
    def edge_dim(self):
        return 50

    @property
    def num_classes(self):
        return 1

    @property
    def deg(self):
        # Compute in-degree histogram over training data.
        deg = torch.zeros(5, dtype=torch.long)
        if self.dataset_train is not None:
            for data in self.dataset_train:
                d = degree(
                    data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long
                )
                deg += torch.bincount(d, minlength=deg.numel())
        else:
            deg = torch.tensor([[0, 41130, 117278, 70152, 3104]])
        return deg.numpy().tolist()

    @property
    def hyper_parameters(self):
        return {
            "num_features": self.num_features,
            "num_classes": self.num_classes,
            "node_dim": self.node_dim,
            "edge_dim": self.edge_dim,
            "deg": self.deg,
        }

    def prepare_data(self):
        path = osp.join(
            osp.dirname(osp.realpath(__file__)), "..", "..", "data", self.NAME
        )

        self.train_dataset = ZINC(path, subset=True, split="train")
        self.val_dateset = ZINC(path, subset=True, split="val")
        self.test_dataset = ZINC(path, subset=True, split="test")

    def training_step(self, batch, batch_nb):
        out = self.forward(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = (out.squeeze() - batch.y).abs().mean()
        result = pl.TrainResult(loss)
        result.log("train_loss", loss, prog_bar=True)
        return result

    def _step(self, batch, batch_nb, stage=None):
        preds = self.forward(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = (preds.squeeze() - batch.y).abs().mean()
        result = pl.EvalResult(checkpoint_on=loss)
        result.log(f"{stage}_loss", loss)
        result.y = batch.y
        result.preds = preds
        result.num_graphs = torch.tensor([batch.num_graphs]).float()
        return result

    def _epoch_end_step(self, outputs, stage=None):
        avg_loss = outputs[f"{stage}_loss"].mean()
        result = pl.EvalResult(checkpoint_on=avg_loss)
        result.log(f"{stage}_loss", avg_loss, prog_bar=True)
        return result

    def validation_step(self, batch, batch_nb):
        return self._step(batch, batch_nb, stage="val")

    def validation_epoch_end(self, outputs):
        return self._epoch_end_step(outputs, stage="val")

    def test_step(self, batch, batch_nb):
        return self._step(batch, batch_nb, stage="test")

    def test_epoch_end(self, outputs):
        return self._epoch_end_step(outputs, stage="test")
