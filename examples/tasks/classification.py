import os
import os.path as osp
import numpy as np
from functools import partial
import numpy as np
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
from examples.tasks.bases import BaseStepsMixin

class BaseClassificationStepsMixin(BaseStepsMixin):
    def __init__(self, *args, **kwargs):
        super(BaseClassificationStepsMixin, self).__init__()

    def compute_result(self, loss, preds, targets, stage):
        if not hasattr(self, "_acc_meter"):
            self._acc_meter = metrics.Accuracy(self.num_classes)
        acc = self.compute_acc(preds, targets)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log(f"{stage}_loss", loss, prog_bar=True)
        result.log(f"{stage}_acc", acc, prog_bar=True)
        return result

class BinaryClassificationStepsMixin(BaseClassificationStepsMixin):

    loss_op = torch.nn.BCEWithLogitsLoss()
    num_classes = 2

    def compute_loss(self, logits, targets, internal_losses):
        loss = self.loss_op(logits.squeeze(), targets.squeeze()) + internal_losses
        return loss, logits

    def compute_acc(self, preds, targets):
        return self._acc_meter((F.sigmoid(preds) > self._threshold), targets.flatten())

class CategoricalClassificationStepsMixin(BaseClassificationStepsMixin):
    def compute_loss(self, logits, targets, internal_losses):
        preds = F.log_softmax(logits, dim=-1)
        loss = F.nll_loss(preds, targets) + internal_losses
        return loss, preds

    def compute_acc(self, preds, targets):
        return self._acc_meter(preds, targets)
