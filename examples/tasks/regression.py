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
from examples.tasks.bases import BaseStepsMixin

class BaseRegressionStepsMixin(BaseStepsMixin):
    def __init__(self, *args, **kwargs):
        super(BaseRegressionStepsMixin, self).__init__()

    def compute_loss(self, inputs, targets, internal_losses):
        loss = self.loss(inputs.squeeze(), targets.squeeze()) + internal_losses
        return loss, inputs

    def compute_result(self, loss, preds, targets, stage):
        result = pl.EvalResult(checkpoint_on=loss)
        result.log(f"{stage}_loss", loss, prog_bar=True)
        return result

class L1RegressionStepsMixin(BaseRegressionStepsMixin):
    def __init__(self, *args, **kwargs):
        super(L1RegressionStepsMixin, self).__init__(*args, **kwargs)

        self.loss = torch.nn.L1Loss()

class L2RegressionStepsMixin(BaseRegressionStepsMixin):
    def __init__(self, *args, **kwargs):
        super(L2RegressionStepsMixin, self).__init__(*args, **kwargs)

        self.loss = torch.nn.MSELoss()
