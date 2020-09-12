import os
import os.path as osp
import numpy as np
from functools import partial
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import LightningDataModule
import pytorch_lightning as pl
import torch_geometric
from torch_geometric.datasets import Reddit
from torch_geometric.data import NeighborSampler
import torch_geometric.transforms as T
from examples.datasets.base_dataset import BaseDataset


class RedditDataset(BaseDataset):

    NAME = "Reddit"

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    @property
    def num_features(self):
        return 1433  # TODO Find a better way to infer it

    @property
    def num_classes(self):
        return 7

    @property
    def hyper_parameters(self):
        return {"num_features": self.num_features, "num_classes": self.num_classes}

    def prepare_data(self):
        path = osp.join(
            osp.dirname(osp.realpath(__file__)), "..", "..", "data", self.NAME
        )
        dataset = Reddit(path)
        self.data = dataset[0]

    def train_dataloader(self, batch_size=32, transforms=None):
        return NeighborSampler(
            self.data.edge_index,
            node_idx=self.data.train_mask,
            sizes=[25, 10],
            batch_size=batch_size,
            shuffle=True,
            num_workers=self._num_workers,
        )

    def val_dataloader(self, batch_size=32, transforms=None):
        return NeighborSampler(
            self.data.edge_index,
            node_idx=self.data.val_mask,
            sizes=[25, 10],
            batch_size=batch_size,
            shuffle=True,
            num_workers=self._num_workers,
        )

    def test_dataloader(self, batch_size=32, transforms=None):
        return NeighborSampler(
            self.data.edge_index,
            node_idx=self.data.test_mask,
            sizes=[25, 10],
            batch_size=batch_size,
            shuffle=True,
            num_workers=self._num_workers,
        )

    def training_step(self, batch, batch_nb):
        loss = F.nll_loss(
            F.log_softmax(self.forward(self.data.x[batch[1]], batch[2]), -1),
            self.data.y[batch[1]],
        )
        result = pl.TrainResult(loss)
        result.log("train_loss", loss)
        return result