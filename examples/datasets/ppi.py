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
import pytorch_lightning as pl
from examples.tasks.classification import BinaryDatasetSteps

from examples.core.base_dataset import BaseDataset


class PPIDataset(BaseDataset, BinaryDatasetSteps):

    NAME = "PPI"

    def __init__(
        self,
        *args,
        threshold=0.5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self._threshold = threshold

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
        self.train_dataset = PPI(path, split="train")
        self.val_dataset = PPI(path, split="val")
        self.test_dataset = PPI(path, split="test")
