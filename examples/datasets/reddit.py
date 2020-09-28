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
from examples.core.base_dataset import BaseDataset
from examples.tasks.classification import CategoricalDatasetSteps


class RedditDataset(BaseDataset, CategoricalDatasetSteps):

    NAME = "Reddit"

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    @property
    def num_features(self):
        return 602  # TODO Find a better way to infer it

    @property
    def num_classes(self):
        return 41

    @property
    def hyper_parameters(self):
        return {"num_features": self.num_features, "num_classes": self.num_classes}

    def prepare_data(self):
        path = osp.join(
            osp.dirname(osp.realpath(__file__)), "..", "..", "data", self.NAME
        )
        self.dataset = Reddit(path)
        self.data = self.dataset[0]