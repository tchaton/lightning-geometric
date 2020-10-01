import os
import re
from functools import partial
import torch
import torch.nn.functional as F
import copy
from omegaconf import OmegaConf
from hydra.utils import instantiate
from os.path import join, dirname
from torch_geometric.data import DataLoader
from pytorch_lightning.utilities.model_utils import is_overridden
from torch_geometric.nn import GNNExplainer
from torch_geometric.nn.conv.utils.jit import class_from_module_repr
from examples.core.base_dataset_samplers import SAMPLING
import logging
from tqdm import tqdm

log = logging.getLogger(__name__)

ROOT_DIR = dirname(dirname(dirname(__file__)))


class CustomGNNExplainer(GNNExplainer):
    def explain_node(self, node_idx, batch, **kwargs):
        r"""Learns and returns a node feature mask and an edge mask that play a
        crucial role to explain the prediction made by the GNN for node
        :attr:`node_idx`.
        Args:
            node_idx (int): The node to explain.
            x (Tensor): The node feature matrix.
            edge_index (LongTensor): The edge indices.
            **kwargs (optional): Additional arguments passed to the GNN module.
        :rtype: (:class:`Tensor`, :class:`Tensor`)
        """

        self.model.eval()
        self.__clear_masks__()

        x = batch.x
        edge_index = batch.edge_index

        num_edges = edge_index.size(1)

        # Only operate on a k-hop subgraph around `node_idx`.
        x, edge_index, mapping, hard_edge_mask, kwargs = self.__subgraph__(
            node_idx, x, edge_index, **kwargs
        )

        # Get the initial prediction.
        with torch.no_grad():
            pred_label = F.log_softmax(self.model(batch)).argmax(dim=-1)

        self.__set_masks__(x, edge_index)
        self.to(x.device)

        optimizer = torch.optim.Adam([self.node_feat_mask, self.edge_mask], lr=self.lr)

        if self.log:  # pragma: no cover
            pbar = tqdm(total=self.epochs)
            pbar.set_description(f"Explain node {node_idx}")

        for epoch in range(1, self.epochs + 1):
            optimizer.zero_grad()
            h = x * self.node_feat_mask.view(1, -1).sigmoid()
            batch.x = h
            batch.edge_index = edge_index
            log_logits = F.log_softmax(self.model(batch))
            loss = self.__loss__(mapping, log_logits, pred_label)
            loss.backward()
            optimizer.step()

            if self.log:  # pragma: no cover
                pbar.update(1)

        if self.log:  # pragma: no cover
            pbar.close()

        node_feat_mask = self.node_feat_mask.detach().sigmoid()
        edge_mask = self.edge_mask.new_zeros(num_edges)
        edge_mask[hard_edge_mask] = self.edge_mask.detach().sigmoid()

        self.__clear_masks__()

        return node_feat_mask, edge_mask


def check_jittable(model, data_module):
    path2model = os.path.join(os.getcwd(), "model.pt")
    torch.jit.save(model.to_torchscript(), path2model)
    model = torch.jit.load(path2model)
    data_module.forward = model.forward
    batch = get_single_batch(data_module)
    _ = data_module.step(batch, None, "test")
    print("Check model is jittable complete.")


def instantiate_data_module(cfg):
    defaulTasksMixin = None
    if hasattr(cfg.model, "params"):
        if hasattr(cfg.model.params, "defaulTasksMixin"):
            # Override dataset mixin
            # TODO Better handle defaulTasksMixin injection
            return instantiate(
                cfg.dataset, defaulTasksMixin=cfg.model.params.defaulTasksMixin
            )
    return instantiate(cfg.dataset)


def run_explainer(explainer, trainer, model, data_module):
    class ExplainerModel:
        def __init__(self, model, data_module):
            self.model = model
            self.data_module = data_module

        def modules(self):
            return self.model.modules()

        def eval(self):
            self.model.eval()

        def __call__(self, batch):
            # return loss, preds, targets
            return self.data_module.inference_step(batch, None, "test")[0][0]

    explainer = CustomGNNExplainer(ExplainerModel(model, data_module), **explainer)
    batch = get_single_batch(data_module)
    node_idx = 0
    node_feat_mask, edge_mask = explainer.explain_node(node_idx, batch.clone())
    breakpoint()
    ax, G = explainer.visualize_subgraph(
        node_idx, batch.edge_index, edge_mask, y=batch.y
    )
    plt.show()


def instantiate_model(cfg, data_module):
    cfg_copy = copy.deepcopy(cfg)
    model: pl.LightningModule = instantiate(
        cfg.model,
        optimizers=cfg.optimizers.optimizers,
        **data_module.hyper_parameters,
    )
    if not cfg.jit:
        attach_step_and_epoch_functions(model, data_module)
        return model
    model_cls = re_compile_model(model, data_module)
    model = model_cls(**dict(model.hparams))
    model.convert_to_jittable()
    attach_step_and_epoch_functions(model, data_module)
    return model


def re_compile_model(model, data_module):
    # is_typed_dataloader = data_module.test_loader_type == SAMPLING.DataLoader.value
    batchType = "TensorBatch"
    path2model = model.__class__.__module__.replace(".", "/")
    with open(join(ROOT_DIR, f"{path2model}.py")) as f:
        code = f.read()
    code = code.replace("batch)", f"batch:{batchType})")
    code = "from examples.core.typing import *\n" + code
    return class_from_module_repr(model.__class__.__name__, code)


def get_single_batch(data_module):
    for batch in data_module.test_dataloader():
        return batch


def attach_step_and_epoch_functions(model, datamodule):
    datamodule.forward = model.forward
    for attr in dir(datamodule):
        if sum([token in attr for token in ["_step", "_epoch_end"]]) > 0:
            if not is_overridden(attr, model):
                setattr(model, attr, getattr(datamodule, attr))