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

from examples.datasets.base_dataset import BaseDataset


class PPIDataset(BaseDataset):

    NAME = "PPI"

    def __init__(
        self,
        seed=42,
        num_workers=4,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._seed = seed
        self._num_workers = num_workers

    @property
    def num_features(self):
        return 50  # TODO Find a better way to infer it

    @property
    def num_classes(self):
        return 121

    def prepare_data(self):
        path = osp.join(
            osp.dirname(osp.realpath(__file__)), "..", "..", "data", self.NAME
        )
        self.dataset_train = PPI(path, split="train")
        self.dataset_val = PPI(path, split="val")
        self.dataset_test = PPI(path, split="test")
