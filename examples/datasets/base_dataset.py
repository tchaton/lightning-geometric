import os
import os.path as osp
import numpy as np
from functools import partial
from hydra.utils import instantiate
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

    NAME = ...

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        self.__instantiate_transform(kwargs)
        super().__init__(*args, **kwargs)

        self.dataset_train = None
        self.dataset_val = None
        self.dataset_test = None

        self._seed = 42
        self._num_workers = 2
        self._hyper_parameters = {}

    def __instantiate_transform(self, kwargs):
        self._pre_transform = None
        self._train_transform = None
        self._val_transform = None
        self._test_transform = None

        for k in [k for k in kwargs]:
            if "transform" in k and kwargs.get(k) is not None:
                transform = T.Compose([instantiate(t) for t in kwargs.get(k)])
                setattr(self, f"_{k}", transform)
                del kwargs[k]

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
            self.dataset_train,
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