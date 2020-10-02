import os
import os.path as osp
import numpy as np
from functools import partial
import torch
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import LightningDataModule
import torch_geometric
from torch_geometric.datasets import ICEWS18
from collections import namedtuple
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import f1_score

from examples.core.base_dataset import BaseDataset
from examples.tasks.bases import BaseNodeStepsMixin


class ICEWS18Dataset(BaseDataset, BaseNodeStepsMixin):

    NAME = "ICEWS18"
    BATCH_KEYS = ["h_obj", "h_obj_t", "h_sub", "h_sub_t", "obj", "rel", "sub", "t"]

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
        return 256

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

        self.train_dataset = ICEWS18(
            path, split="train", pre_transform=self._pre_transform
        )
        self.val_dataset = ICEWS18(path, split="val")
        self.test_dataset = ICEWS18(path, split="test")

    def prepare_batch(self, batch, batch_nb, stage, sampling):
        Batch: TensorBatch = namedtuple("Batch", self.BATCH_KEYS)
        return Batch(**{k: batch[k] for k in batch.keys}), batch.t

    def compute_loss(self, batch):
        log_prob_obj, log_prob_sub = self.forward(batch)
        loss_obj = F.nll_loss(log_prob_obj, batch.obj)
        loss_sub = F.nll_loss(log_prob_sub, batch.sub)
        loss = loss_obj + loss_sub
        return loss, log_prob_obj, log_prob_sub
