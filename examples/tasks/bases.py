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
from torch_geometric.nn import global_mean_pool, global_max_pool
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import (
    v_measure_score,
    homogeneity_score,
    completeness_score,
)
from examples.core.base_dataset_samplers import SAMPLING
from examples.core.typing import SparseBatch, TensorBatch


class GANMode(Enum):
    Encoder = "gen"
    Discriminator = "dis"


class VGAEMode(Enum):
    Encoder = "enc"
    Discriminator = "dis"


class BaseStepsMixin:
    def inference_step(self, batch, batch_nb, stage, generative_mode=None):
        sampling = None
        for sampler in self._samplers:
            if sampler.stage == stage:
                sampling = sampler.sampling
        typed_batch, targets = self.prepare_batch(batch, batch_nb, stage, sampling)
        if typed_batch.x.is_leaf:
            typed_batch.x.requires_grad = True
        if generative_mode is not None:
            outputs = self.forward(
                stage == "train",
                self.GENERATIVE_TYPE.Discriminator == generative_mode,
                typed_batch,
            )
        else:
            outputs = self.forward(typed_batch)

        return outputs, targets, sampling

    def step(self, batch, batch_nb, stage, generative_mode=None):
        if generative_mode is not None:
            (
                (logits, internal_losses, internal_metrics),
                targets,
                sampling,
            ) = self.inference_step(
                batch, batch_nb, stage, generative_mode=generative_mode
            )
        else:
            (logits, internal_losses), targets, sampling = self.inference_step(
                batch, batch_nb, stage
            )
        
        if logits is not None:
            if sampling == SAMPLING.DataLoader.value:
                logits = logits[batch[f"{stage}_mask"]]

        if not any(set(logits.shape).intersection(targets.shape)):
            if batch.batch is not None:
                logits = global_max_pool(logits.squeeze(), batch.batch)
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

        elif (
            sampling == SAMPLING.NeighborSampler.value
            or sampling == SAMPLING.GraphSAINTRandomWalkSampler.value
        ):
            return (
                Batch(
                    self.data.x[batch[1]],
                    [e.edge_index for e in batch[2]],
                    None,
                    None,
                ),
                self.data.y[batch[1]],
            )
        elif sampling == SAMPLING.LinkPred.value:
            Batch = namedtuple("Data", ["x", "pos_edge_index", "neg_edge_index", "pos_edge_attr"])
            return Batch(*[batch.x, batch.pos_edge_index, batch.neg_edge_index, getattr(batch, "pos_edge_attr", None)]), batch.link_labels
        else:
            raise Exception("Not defined")

    def validation_step(self, batch, batch_nb):
        return self._test_step(batch, batch_nb, stage="val")

    def test_step(self, batch, batch_nb):
        return self._test_step(batch, batch_nb, stage="test")

    def training_step(self, batch, batch_nb, sampling=None):
        loss, _, _ = self.step(batch, batch_nb, "train")
        result = pl.TrainResult(loss)
        result.log("train_loss", loss, prog_bar=True)
        return result

    def _test_step(
        self,
        batch,
        batch_nb,
        stage=None,
    ):
        loss, preds, targets = self.step(batch, batch_nb, stage)
        return self.compute_result(loss, preds, targets, stage)


class BaseGenerativeStepsMixin(BaseStepsMixin):

    GENERATIVE_TYPE = ...

    def __init__(self, *args, **kwargs):
        pass

    def compute_loss(self, logits, targets, internal_losses):
        return internal_losses, logits

    def training_step(self, batch, batch_nb, optimizer_idx, sampling=False):
        stage = "train"
        # train generator
        generative_mode = (
            self.GENERATIVE_TYPE.Encoder
            if optimizer_idx == 0
            else self.GENERATIVE_TYPE.Discriminator
        )
        if optimizer_idx == 0:
            loss, preds, targets = self.step(batch, batch_nb, stage, generative_mode)
        # train discriminator
        if optimizer_idx == 1:
            loss, preds, targets = self.step(batch, batch_nb, stage, generative_mode)
        result = pl.TrainResult(minimize=loss)
        result.log(f"{generative_mode.value}_{stage}_loss", loss, prog_bar=True)
        return result

    def _test_step(self, batch, batch_nb, stage=None):
        generative_mode = self.GENERATIVE_TYPE.Encoder
        loss, preds, targets = self.step(batch, batch_nb, stage, generative_mode)
        return self.compute_result(loss, preds, targets, stage)

    def compute_result(self, loss, preds, targets, stage):
        # Cluster embedded values using k-means.
        kmeans_input = preds.cpu().numpy()
        kmeans = KMeans(n_clusters=7, random_state=0).fit(kmeans_input)
        pred = kmeans.predict(kmeans_input)

        labels = targets.cpu().numpy()
        completeness = torch.Tensor([completeness_score(labels, pred)])
        hm = torch.Tensor([homogeneity_score(labels, pred)])
        nmi = torch.Tensor([v_measure_score(labels, pred)])

        # auc, ap = model.test(z, data.test_pos_edge_index, data.test_neg_edge_index)
        result = pl.EvalResult(loss)
        result.log(f"{stage}_completeness", completeness, prog_bar=True)
        result.log(f"{stage}_hm", hm, prog_bar=True)
        result.log(f"{stage}_nmi", nmi, prog_bar=True)
        return result


class BaseGanStepsMixin(BaseGenerativeStepsMixin):

    GENERATIVE_TYPE = GANMode


class BaseVgaeStepsMixin(BaseGenerativeStepsMixin):

    GENERATIVE_TYPE = VGAEMode