import os
import os.path as osp
import numpy as np
from functools import partial
from hydra.utils import instantiate
import torch
from torch.utils.data import random_split
from torch_geometric.data import DataLoader
from pytorch_lightning import LightningDataModule
import torch_geometric
from torch_geometric.datasets import PPI
import torch_geometric.transforms as T
from examples.core.base_dataset_samplers import BaseDatasetSampler


def del_attr(kwargs, name):
    try:
        del kwargs[name]
    except:
        pass


class BaseDataset(BaseDatasetSampler, LightningDataModule):

    NAME = ...

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        self.__instantiate_transform(kwargs)
        BaseDatasetSampler.__init__(self, *args, **kwargs)
        self.clean_kwargs(kwargs)
        LightningDataModule.__init__(self, *args, **kwargs)

        self.dataset_train = None
        self.dataset_val = None
        self.dataset_test = None

        self._seed = 42
        self._num_workers = 2
        self._shuffle = True
        self._drop_last = False
        self._pin_memory = True
        self._follow_batch = []

        self._hyper_parameters = {}

    def clean_kwargs(self, kwargs):
        del_attr(kwargs, "samplers")
        del_attr(kwargs, "num_edges")
        del_attr(kwargs, "num_layers")

    @property
    def config(self):
        return {"dataset_config": {}}

    def __instantiate_transform(self, kwargs):
        self._pre_transform = None
        self._transform = None
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