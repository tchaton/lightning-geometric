import os
import os.path as osp
import numpy as np
from functools import partial
import torch
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import LightningDataModule
import torch_geometric
from torch_geometric.datasets import ZINC
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import f1_score
from torch_geometric.utils import degree
from examples.core.base_dataset import BaseDataset
from examples.tasks.regression import NodeRegressionStepsMixin


class ZINCDataset(BaseDataset, NodeRegressionStepsMixin):

    NAME = "ZINC"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def node_dim(self):
        return 75

    @property
    def edge_dim(self):
        return 50

    @property
    def num_classes(self):
        return 1

    @property
    def deg(self):
        # Compute in-degree histogram over training data.
        deg = torch.zeros(5, dtype=torch.long)
        if self.dataset_train is not None:
            for data in self.dataset_train:
                d = degree(
                    data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long
                )
                deg += torch.bincount(d, minlength=deg.numel())
        else:
            deg = torch.tensor([[0, 41130, 117278, 70152, 3104]])
        return deg.numpy().tolist()

    @property
    def hyper_parameters(self):
        return {
            "num_features": self.node_dim,
            "num_classes": self.num_classes,
            "node_dim": self.node_dim,
            "edge_dim": self.edge_dim,
            "deg": self.deg,
        }

    def prepare_data(self):
        path = osp.join(
            osp.dirname(osp.realpath(__file__)), "..", "..", "data", self.NAME
        )

        self.train_dataset = ZINC(path, subset=True, split="train")
        self.val_dataset = ZINC(path, subset=True, split="val")
        self.test_dataset = ZINC(path, subset=True, split="test")