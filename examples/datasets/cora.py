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
from examples.tasks.classification import ClassificationDatasetHook
from examples.core.base_dataset import BaseDataset


class CoraDataset(BaseDataset, ClassificationDatasetHook):

    NAME = "cora"

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

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
