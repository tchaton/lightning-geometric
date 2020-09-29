import os
import re
import torch
import copy
from hydra.utils import instantiate
from os.path import join, dirname
from torch_geometric.data import DataLoader
from pytorch_lightning.utilities.model_utils import is_overridden
from torch_geometric.nn.conv.utils.jit import class_from_module_repr
from examples.core.base_dataset_samplers import SAMPLING

ROOT_DIR = dirname(dirname(dirname(__file__)))


def check_jittable(model, data_module):
    path2model = os.path.join(os.getcwd(), "model.pt")
    torch.jit.save(model.to_torchscript(), path2model)
    model = torch.jit.load(path2model)
    data_module.forward = model.forward
    batch = get_single_batch(data_module)
    _ = data_module.step(get_single_batch(data_module), None, "test")
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
    model = model_cls(
        **cfg_copy.model.params,
        optimizers=cfg_copy.optimizers.optimizers,
        **data_module.hyper_parameters,
    )
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