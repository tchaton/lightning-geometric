from dataclasses import dataclass, field
from typing import *
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, MISSING

defaults = [
    # An error will be raised if the user forgets to specify them`
    {"model": MISSING},
    {"dataset": MISSING},
    {"optimizers": "adam"},
    {"trainer": "debugging"},
]


@dataclass
class Trainer:
    max_epochs: int = 100
    gpus: int = 0


@dataclass
class ObjectConf(Dict[str, Any]):
    # class, class method or function name
    _target_: str = MISSING
    # parameters to pass to target when calling it
    params: Any = field(default_factory=dict)


@dataclass
class OptimizerConf(Dict[str, Any]):
    # class, class method or function name
    optimizers: List = MISSING


@dataclass
class Config:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    # Hydra will populate this field based on the defaults list
    model: Any = MISSING
    dataset: Any = MISSING
    optimizers: Any = MISSING
    trainer: Any = MISSING


cs = ConfigStore.instance()
cs.store(name="config", node=Config)
cs.store(group="model", name="agnn", node=ObjectConf)
cs.store(group="dataset", name="cora", node=ObjectConf)
cs.store(group="optimizers", name="adam", node=OptimizerConf)
cs.store(group="trainer", name="debugging", node=ObjectConf)