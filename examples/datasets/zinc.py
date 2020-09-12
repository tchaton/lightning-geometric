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

from examples.datasets.base_dataset import BaseDataset


class ZINCDataset(BaseDataset):

    NAME = "ZINC"

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )

    @property
    def node_dim(self):
        return 75

    @property
    def edge_dim(self):
        return 50

    @property
    def num_features(self):
        return 3  # TODO Find a better way to infer it

    @property
    def num_classes(self):
        return 6890

    @property
    def deg(self):
        # Compute in-degree histogram over training data.
        deg = torch.zeros(5, dtype=torch.long)
        if self.dataset_train is not None:
            for data in train_dataset:
                d = degree(
                    data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long
                )
                deg += torch.bincount(d, minlength=deg.numel())
        return deg.numpy().tolist()

    @property
    def hyper_parameters(self):
        return {
            "num_features": self.num_features,
            "num_classes": self.num_classes,
            "node_dim": self.node_dim,
            "edge_dim": self.edge_dim,
            "deg": self.deg,
        }

    def prepare_data(self):
        path = osp.join(
            osp.dirname(osp.realpath(__file__)), "..", "..", "data", self.NAME
        )

        self.dataset_train = ZINC(path, subset=True, split="train")
        self.dataset_val = ZINC(path, subset=True, split="val")
        self.dataset_test = ZINC(path, subset=True, split="test")

    def training_step(self, batch, batch_nb):
        out = self.forward(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = (out.squeeze() - batch.y).abs().mean()
        import pdb

        pdb.set_trace()
        result = pl.TrainResult(loss)
        result.log("train_loss", loss, prog_bar=True)
        return result

    def test_step(self, batch, batch_nb):
        preds = self.forward(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = (out.squeeze() - batch.y).abs().mean()
        result = pl.EvalResult(loss)
        result.log("test_loss", loss)
        result.y = batch.y
        result.preds = preds
        result.num_graphs = torch.tensor([batch.num_graphs]).float()
        return result

    def test_epoch_end(self, outputs):
        avg_loss = outputs["test_loss"].sum() / outputs["num_graphs"].sum()
        test_f1_score = torch.tensor(
            [f1_score(outputs["y"], outputs["preds"] > 0, average="micro")]
        )
        result = pl.EvalResult(checkpoint_on=test_f1_score)
        result.log("test_loss", avg_loss, prog_bar=True)
        result.log("test_f1_score", test_f1_score, prog_bar=True)
        return result
