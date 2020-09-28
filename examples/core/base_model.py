import os.path as osp
from omegaconf import OmegaConf, DictConfig, ListConfig
from functools import partial
from hydra.utils import instantiate, get_class
import torch
from torch import nn
from torch.nn import Sequential, ModuleDict, ModuleList
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, MessagePassing
import pytorch_lightning as pl
from examples.core.typing import *


class BaseModel(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self._optimizers_conf = self._validate_optimizers_conf(kwargs["optimizers"])
        self._optimizer_name = None
        if len(self._optimizers_conf) == 1:
            self._optimizer_name = self._optimizers_conf[0]["name"]
        self._init_optim = partial(
            self._init_optim,
            optimizers_conf=self._optimizers_conf,
        )

    @property
    def config(self):
        config = {"model_config": {}}
        config["model_config"].update(
            {k: v for k, v in self.hparams.items() if isinstance(v, (int, float, str))}
        )
        for optim_conf in self._optimizers_conf:
            optim_dict = {f"optim_{optim_conf.name}_class": optim_conf._target_}
            for (param_name, param_value) in optim_conf.params.items():
                optim_dict[f"optim_{optim_conf.name}_{param_name}"] = param_value

            config["model_config"].update(optim_dict)
        return config

    @staticmethod
    def _validate_optimizers_conf(optimizers):
        names = []
        for optim_conf in optimizers:
            not_name = not hasattr(optim_conf, "name")
            not_target = not hasattr(optim_conf, "_target_")
            not_params = not hasattr(optim_conf, "params")
            if not_name or not_target or not_params:
                msg = ""
                if not_name:
                    msg += f"name should be within {optim_conf} "
                if not_target:
                    msg += f"_target_ should be within {optim_conf} "
                if not_params:
                    msg += f"params should be within {optim_conf} "
                raise Exception(f"{msg}")
            if not not_name:
                names.append(optim_conf.name)
        if len(names) != len(set(names)):
            raise Exception(
                f"Each optimizer name should be unique. Here is the list of optimizer names: {names}"
            )
        return optimizers

    @staticmethod
    def _init_optim(name, params, optimizers_conf=None):
        if optimizers_conf is not None:
            for optim_conf in optimizers_conf:
                if name == optim_conf["name"]:
                    optim_cls = get_class(optim_conf["_target_"])
                    return optim_cls([p for p in params], **optim_conf["params"])
            raise Exception(
                f"The provided name {name} doesn't exist within {[o['name'] for o in optimizers_conf]}"
            )
        else:
            raise Exception("Optimizer should be defined within configuration files")

    def configure_optimizers(self):
        if self._optimizer_name is not None:
            return self._init_optim(self._optimizer_name, self.parameters())
        else:
            raise Exception(
                "Multiple optimizers are defined. Please, override configure_optimizers function"
            )

    def _convert_to_jittable(self, module):
        for key, m in module._modules.items():
            if isinstance(m, MessagePassing):
                module._modules[key] = m.jittable()
            else:
                self._convert_to_jittable(m)
        return module

    def convert_to_jittable(self):
        for key, m in self._modules.items():
            self._modules[key] = self._convert_to_jittable(m)

    @torch.jit._overload_method
    def forward(self, batch: SparseBatch):
        pass

    @torch.jit._overload_method
    def forward(self, batch: TensorBatch):
        pass

    def sanetize_kwargs(self, kwargs):
        def _sanetize_kwargs(value):
            if isinstance(value, (DictConfig, ListConfig)):
                return OmegaConf.to_container(value)
            elif isinstance(value, list):
                return [_sanetize_kwargs(v) for v in value]
            elif isinstance(value, dict):
                return {k: _sanetize_kwargs(v) for k, v in value.items()}
            else:
                return value

        for key, value in kwargs.items():
            kwargs[key] = _sanetize_kwargs(value)
        return kwargs
