import os
import os.path as osp
import numpy as np
from functools import partial
import torch
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import LightningDataModule
import torch_geometric
from torch_geometric.datasets import FAUST
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import f1_score
from examples.core.base_dataset import BaseDataset
from examples.tasks.classification import CategoricalDatasetSteps
from examples.core.base_dataset import BaseDataset


class FAUSTDataset(BaseDataset, CategoricalDatasetSteps):

    NAME = "FAUST"

    def __init__(
        self,
        *args,
        pre_transform=None,
        train_transform=None,
        test_transform=None,
        **kwargs,
    ):
        super().__init__(
            *args,
            pre_transform=pre_transform,
            train_transform=train_transform,
            test_transform=test_transform,
            **kwargs,
        )

    @property
    def num_features(self):
        return 3  # TODO Find a better way to infer it

    @property
    def num_classes(self):
        return 6890

    def prepare_data(self):
        path = osp.join(
            osp.dirname(osp.realpath(__file__)), "..", "..", "data", self.NAME
        )

        self.dataset_train = FAUST(
            path, True, self._train_transform, self._pre_transform
        )
        self.dataset_test = FAUST(
            path, False, self._test_transform, self._pre_transform
        )

    def training_step(self, batch, batch_nb):

        loss = F.nll_loss(
            F.log_softmax(
                self.forward(batch.x, batch.edge_index, batch.edge_attr), dim=1
            ),
            batch.y,
        )
        result = pl.TrainResult(loss)
        result.log("train_loss", loss, prog_bar=True)
        return result

    def test_step(self, batch, batch_nb):
        preds = F.log_softmax(
            self.forward(batch.x, batch.edge_index, batch.edge_attr), dim=1
        )
        loss = F.nll_loss(
            preds,
            batch.y,
        )
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
