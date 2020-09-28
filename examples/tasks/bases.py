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


class BaseDatasetSteps:
    def step(self, batch, batch_nb, stage):
        sampling = None
        for sampler in self._samplers:
            if sampler.stage == stage:
                sampling = sampler.sampling
        typed_batch, targets = self.prepare_batch(batch, batch_nb, stage, sampling)
        logits, internal_losses = self.forward(typed_batch)
        if sampling == SAMPLING.DataLoader.value:
            logits = logits[batch[f"{stage}_mask"]]
        loss, preds = self.compute_loss(logits, targets, internal_losses)
        return loss, preds, targets

    def prepare_batch(self, batch, batch_nb, stage, sampling):
        if sampling == SAMPLING.DataLoader.value:
            stage_mask = batch[f"{stage}_mask"]
            edges_keys = [k for k in batch.keys if "edge" in k]
            node_keys = sorted([k for k in batch.keys if k in ["x", "batch", "y"]])
            keys = edges_keys + node_keys[:-1]
            Batch = namedtuple("TensorBatch", keys)
            batch_args = {}
            for key in keys:
                batch_args[key] = (
                    batch[key][stage_mask]
                    if stage_mask is not None and key in node_keys
                    else batch[key]
                )
                if key == "edge_index":
                    batch_args[key] = [batch_args[key].long()]
            targets = batch["y"][stage_mask] if stage_mask is not None else batch["y"]
            return Batch(**batch_args), targets
        elif sampling == SAMPLING.NeighborSampler.value:
            Batch = namedtuple("SparseBatch", ["x", "edge_index"])
            return (
                Batch(
                    self.data.x[batch[1]],
                    [e.edge_index for e in batch[2]],
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

        acc = self.compute_acc(preds, targets)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log(f"{stage}_loss", loss, prog_bar=True)
        result.log(f"{stage}_acc", acc, prog_bar=True)
        return result

    def validation_step(self, batch, batch_nb):
        return self._test_step(batch, batch_nb, stage="val")

    def test_step(self, batch, batch_nb):
        return self._test_step(batch, batch_nb, stage="test")