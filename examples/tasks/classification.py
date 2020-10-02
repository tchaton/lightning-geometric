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
from collections import namedtuple
from torch_geometric.data import DataLoader
from torch_geometric.datasets import Planetoid
from torch_geometric.data import NeighborSampler
import torch_geometric.transforms as T
from examples.core.base_dataset_samplers import SAMPLING
from examples.core.typing import SparseBatch, TensorBatch
from examples.tasks.bases import BaseNodeStepsMixinMixin


class BaseNodeClassificationStepsMixin(BaseNodeStepsMixinMixin):
    def __init__(self, *args, **kwargs):
        super(BaseNodeClassificationStepsMixin, self).__init__()

    def compute_result(self, loss, preds, targets, stage):
        if not hasattr(self, "_acc_meter"):
            self._acc_meter = metrics.Accuracy(self.num_classes)
        acc = self.compute_acc(preds, targets)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log(f"{stage}_loss", loss, prog_bar=True)
        result.log(f"{stage}_acc", acc, prog_bar=True)
        return result


class NodeCategoricalClassificationStepsMixin(BaseNodeClassificationStepsMixin):
    def compute_loss(self, logits, targets, internal_losses):
        preds = F.log_softmax(logits, -1)
        loss = F.nll_loss(preds, targets) + internal_losses
        return loss, preds

    def compute_acc(self, preds, targets):
        return self._acc_meter(preds, targets)


class NodeBinaryClassificationStepsMixin(BaseNodeClassificationStepsMixin):

    loss_op = torch.nn.BCEWithLogitsLoss()

    def compute_loss(self, logits, targets, internal_losses):
        preds = F.sigmoid(logits)
        loss = self.loss_op(preds.squeeze(), targets) + internal_losses
        return loss, preds

    def compute_acc(self, preds, targets):
        return self._acc_meter((preds > self._threshold), targets.flatten())