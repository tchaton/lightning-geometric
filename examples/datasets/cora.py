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
from torch_geometric.datasets import Planetoid
from torch_geometric.data import NeighborSampler
import torch_geometric.transforms as T

dataloader = partial(
    torch.utils.data.DataLoader,
    collate_fn=torch_geometric.data.batch.Batch.from_data_list,
    worker_init_fn=np.random.seed,
)


class CoraDataset(LightningDataModule):

    NAME = "cora"

    def __init__(
        self,
        val_split,
        transforms,
        seed=42,
        num_workers=2,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._val_split = val_split
        self._seed = seed
        self._num_workers = num_workers

    @property
    def num_features(self):
        return 1433  # TODO Find a better way to infer it

    @property
    def num_classes(self):
        return 7

    @property
    def hyper_parameters(self):
        return {"num_features": self.num_features, "num_classes": self.num_classes}

    @staticmethod
    def _extract_index_edges(edge_index, mask):
        edge_mask = mask.nonzero().squeeze().unsqueeze(0).repeat((2, 1))
        edge_index = torch.gather(edge_index, 1, edge_mask)
        return edge_index - edge_index.min()

    def prepare_data(self):
        path = osp.join(
            osp.dirname(osp.realpath(__file__)), "..", "..", "data", self.NAME
        )
        dataset = Planetoid(path, self.NAME, transform=T.NormalizeFeatures())
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