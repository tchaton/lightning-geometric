import os
import os.path as osp
from enum import Enum
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


class GANMode(Enum):
    Generator = "gen"
    Discriminator = "dis"


class BaseDatasetStepsMixin:
    def step(self, batch, batch_nb, stage):
        sampling = None
        for sampler in self._samplers:
            if sampler.stage == stage:
                sampling = sampler.sampling
        typed_batch, targets = self.prepare_batch(batch, batch_nb, stage, sampling)
        logits, internal_losses = self.forward(typed_batch)
        if logits is not None:
            if sampling == SAMPLING.DataLoader.value:
                logits = logits[batch[f"{stage}_mask"]]
        loss, preds = self.compute_loss(logits, targets, internal_losses)
        return loss, preds, targets

    def prepare_batch(self, batch, batch_nb, stage, sampling):
        usual_keys = ["x", "edge_index", "edge_attr", "batch"]
        Batch: TensorBatch = namedtuple("Batch", usual_keys)
        if sampling == SAMPLING.DataLoader.value:
            stage_mask = batch[f"{stage}_mask"]
            batch_args = {}
            for key in usual_keys:
                if key in batch.keys:
                    batch_args[key] = batch[key]
                    if key == "edge_index":
                        batch_args[key] = [
                            batch_args[key].long() for _ in range(self._num_layers)
                        ]
                else:
                    batch_args[key] = None
            targets = batch["y"][stage_mask] if stage_mask is not None else batch["y"]
            return Batch(**batch_args), targets

        elif sampling == SAMPLING.NeighborSampler.value:
            return (
                Batch(
                    self.data.x[batch[1]],
                    [e.edge_index for e in batch[2]],
                    None,
                    None,
                ),
                self.data.y[batch[1]],
            )
        else:
            raise Exception("Not defined")

    def training_step(self, batch, batch_nb, sampling=None):
        loss, _, _ = self.step(batch, batch_nb, "train")
        result = pl.TrainResult(loss)
        result.log("train_loss", loss, prog_bar=True)
        return result

    def _test_step(self, batch, batch_nb, stage=None):
        loss, preds, targets = self.step(batch, batch_nb, stage)
        return self.compute_result(loss, preds, targets, stage)

    def validation_step(self, batch, batch_nb):
        return self._test_step(batch, batch_nb, stage="val")

    def test_step(self, batch, batch_nb):
        return self._test_step(batch, batch_nb, stage="test")


class BaseGANDatasetStepsMixin:
    def __init__(self, *args, **kwargs):
        pass

    def step(self, batch, batch_nb, stage, gan_mode):
        sampling = None
        for sampler in self._samplers:
            if sampler.stage == stage:
                sampling = sampler.sampling
        typed_batch, targets = self.prepare_batch(batch, batch_nb, stage, sampling)
        logits, internal_losses = self.forward(
            GANMode.Discriminator == gan_mode, typed_batch
        )
        if logits is not None:
            if sampling == SAMPLING.DataLoader.value:
                logits = logits[batch[f"{stage}_mask"]]
        loss, preds = self.compute_loss(logits, targets, internal_losses)
        return loss, preds, targets

    def compute_loss(self, logits, targets, internal_losses):
        return internal_losses, logits

    def prepare_batch(self, batch, batch_nb, stage, sampling):
        usual_keys = ["x", "edge_index", "edge_attr", "batch"]
        Batch: TensorBatch = namedtuple("Batch", usual_keys)
        if sampling == SAMPLING.DataLoader.value:
            stage_mask = batch[f"{stage}_mask"]
            batch_args = {}
            for key in usual_keys:
                if key in batch.keys:
                    batch_args[key] = batch[key]
                    if key == "edge_index":
                        batch_args[key] = [
                            batch_args[key].long() for _ in range(self._num_layers)
                        ]
                else:
                    batch_args[key] = None
            targets = batch["y"][stage_mask] if stage_mask is not None else batch["y"]
            return Batch(**batch_args), targets

        elif sampling == SAMPLING.NeighborSampler.value:
            return (
                Batch(
                    self.data.x[batch[1]],
                    [e.edge_index for e in batch[2]],
                    None,
                    None,
                ),
                self.data.y[batch[1]],
            )
        else:
            raise Exception("Not defined")

    def training_step(self, batch, batch_nb, optimizer_idx, sampling=False):
        stage = "train"
        # train generator
        gan_mode = GANMode.Generator if optimizer_idx == 0 else GANMode.Discriminator
        if optimizer_idx == 0:
            loss, preds, targets = self.step(batch, batch_nb, stage, gan_mode)
        # train discriminator
        if optimizer_idx == 1:
            loss, preds, targets = self.step(batch, batch_nb, stage, gan_mode)
        result = pl.TrainResult(minimize=loss)
        result.log(f"{gan_mode.value}_{stage}_loss", loss, prog_bar=True)
        return result