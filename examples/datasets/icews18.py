import os
import os.path as osp
import numpy as np
from functools import partial
import torch
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import LightningDataModule
import torch_geometric
from torch_geometric.datasets import ICEWS18
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import f1_score

from examples.datasets.base_dataset import BaseDataset


class ICEWS18Dataset(BaseDataset):

    NAME = "ICEWS18"

    def __init__(
        self,
        *args,
        seq_len,
        follow_batch,
        pre_transform=None,
        **kwargs,
    ):
        super().__init__(
            *args,
            pre_transform=pre_transform,
            **kwargs,
        )

        self.follow_batch = follow_batch

    @property
    def num_rels(self):
        return 256  # TODO Find a better way to infer it

    @property
    def num_nodes(self):
        return 23033

    @property
    def hyper_parameters(self):
        return {"num_nodes": self.num_nodes, "num_rels": self.num_rels}

    def prepare_data(self):
        path = osp.join(
            osp.dirname(osp.realpath(__file__)), "..", "..", "data", self.NAME
        )

        self.dataset_train = ICEWS18(
            path, split="train", pre_transform=self._pre_transform
        )
        self.dataset_test = ICEWS18(path, split="test")

    def compute_loss(self, batch):
        log_prob_obj, log_prob_sub = self.forward(batch)
        loss_obj = F.nll_loss(log_prob_obj, batch.obj)
        loss_sub = F.nll_loss(log_prob_sub, batch.sub)
        loss = loss_obj + loss_sub
        return loss, log_prob_obj, log_prob_sub

    def training_step(self, batch, batch_nb):
        loss, _, _ = self.compute_loss(batch)
        result = pl.TrainResult(loss)
        result.log("train_loss", loss, prog_bar=True)
        return result

    def test_step(self, batch, batch_nb):
        with torch.no_grad():
            loss, log_prob_obj, log_prob_sub = self.compute_loss(batch)
        result = pl.EvalResult(loss)
        result.log("test_loss", loss)
        return result
