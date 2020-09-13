import os
import os.path as osp
import numpy as np
from functools import partial
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningDataModule
from pytorch_lightning import metrics
import pytorch_lightning as pl
import torch_geometric
from argparse import Namespace
from torch_geometric.data import DataLoader
from torch_geometric.datasets import Planetoid
from torch_geometric.data import NeighborSampler
import torch_geometric.transforms as T
from examples.datasets.base_dataset import BaseDataset


class CoraDataset(BaseDataset):

    NAME = "cora"

    def __init__(
        self,
        num_layers,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self._num_layers = num_layers

    @property
    def num_features(self):
        return 1433

    @property
    def num_classes(self):
        return 7

    def prepare_data(self):
        path = osp.join(
            osp.dirname(osp.realpath(__file__)), "..", "..", "data", self.NAME
        )
        self.dataset = Planetoid(path, self.NAME, transform=self._transform)
        self.data = self.dataset[0]
        self.acc_meter = metrics.Accuracy(self.num_classes)

    def training_step(self, batch, batch_nb):
        preds = F.log_softmax(self.forward(self.data.x[batch[1]], batch[2]), -1)
        loss = F.nll_loss(
            preds,
            self.data.y[batch[1]],
        )
        result = pl.TrainResult(loss)
        result.log("train_loss", loss, prog_bar=True)
        return result

    def _step(self, batch, batch_nb, stage=None):
        if stage is None:
            raise Exception("Stage should be defined")
        stage_mask = batch[f"{stage}_mask"]
        targets = batch.y[stage_mask]
        adjs = [Namespace(edge_index=batch.edge_index) for _ in range(self._num_layers)]
        preds = F.log_softmax(self.forward(batch.x, adjs), -1)[stage_mask]
        loss = F.nll_loss(preds, targets)
        acc = self.acc_meter(preds, targets)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log(f"{stage}_loss", loss, prog_bar=True)
        result.log(f"{stage}_acc", acc, prog_bar=True)
        return result

    def validation_step(self, batch, batch_nb):
        return self._step(batch, batch_nb, stage="val")

    def test_step(self, batch, batch_nb):
        return self._step(batch, batch_nb, stage="test")


class CoraNeighborSamplerDataset(CoraDataset):

    NAME = "cora"

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def train_dataloader(self, batch_size=32, transforms=None):
        return NeighborSampler(
            self.data.edge_index,
            node_idx=self.data.train_mask,
            sizes=[-1, self._num_layers],
            batch_size=batch_size,
            shuffle=True,
            num_workers=self._num_workers,
        )

    def val_dataloader(self, batch_size=1, transforms=None):
        loader = DataLoader(
            self.dataset,
            batch_size=batch_size
            if batch_size <= len(self.dataset)
            else len(self.dataset),
            shuffle=False,
            num_workers=self._num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return loader

    def test_dataloader(self, batch_size=1, transforms=None):
        loader = DataLoader(
            self.dataset,
            batch_size=batch_size
            if batch_size <= len(self.dataset)
            else len(self.dataset),
            shuffle=False,
            num_workers=self._num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return loader