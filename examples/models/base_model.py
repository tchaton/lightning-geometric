import os.path as osp
from omegaconf import OmegaConf
from functools import partial
from hydra.utils import instantiate, get_class
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import pytorch_lightning as pl


class BaseModel(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self._init_optim = partial(self._init_optim, optimizer_conf=kwargs["optimizer"])

    @staticmethod
    def _init_optim(params, optimizer_conf=None):
        if optimizer_conf is not None:
            optim_cls = get_class(optimizer_conf["_target_"])
            return optim_cls([p for p in params], **optimizer_conf["params"])
        else:
            raise Exception("Optimizer should be defined within configuration files")
