import os
import os.path as osp
import numpy as np
from functools import partial
import torch
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import LightningDataModule
import torch_geometric
from torch_geometric.datasets import PPI
import torch_geometric.transforms as T

dataloader = partial(
    torch.utils.data.DataLoader,
    collate_fn=torch_geometric.data.batch.Batch.from_data_list,
    worker_init_fn=np.random.seed,
)


class BaseDataset(LightningDataModule):

    NAME = "PPI"

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    @property
    def num_features(self):
        pass

    @property
    def num_classes(self):
        pass

    @property
    def hyper_parameters(self):
        return {"num_features": self.num_features, "num_classes": self.num_classes}

    def prepare_data(self):
        pass

    def train_dataloader(self, batch_size=1, transforms=None):
        loader = dataloader(
            self.dataset_val,
            batch_size=batch_size
            if batch_size <= len(self.dataset_train)
            else len(self.dataset_train),
            shuffle=True,
            num_workers=self._num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return loader

    def val_dataloader(self, batch_size=1, transforms=None):
        loader = dataloader(
            self.dataset_val,
            batch_size=batch_size
            if batch_size <= len(self.dataset_val)
            else len(self.dataset_val),
            shuffle=False,
            num_workers=self._num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return loader

    def test_dataloader(self, batch_size=1, transforms=None):
        loader = DataLoader(
            self.dataset_test,
            batch_size=batch_size
            if batch_size <= len(self.dataset_test)
            else len(self.dataset_test),
            shuffle=False,
            num_workers=self._num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return loader