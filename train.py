import os
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

os.environ["HYDRA_FULL_ERROR"] = "1"
from omegaconf import DictConfig

import torch
import pytorch_lightning as pl
from examples.config import *
from examples.datasets import *
from examples.core import *
from examples.utils.loggers import initialize_loggers
from examples.utils import (
    instantiate_model,
    instantiate_data_module,
    run_explainer,
    check_jittable,
)
import logging

log = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:
    log.info(OmegaConf.to_yaml(cfg))

    data_module: pl.LightningDataModule = instantiate_data_module(cfg)
    model: pl.LightningModule = instantiate_model(cfg, data_module)

    loggers: List[pl.callbacks.Callback] = initialize_loggers(
        cfg, **model.config, **data_module.config
    )

    gpus = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else None

    resume_from_checkpoint = None

    trainer: pl.Trainer = instantiate(cfg.trainer, gpus=gpus, logger=loggers)

    trainer.fit(model, data_module)
    log.info("Training complete.")

    if cfg.explainer.activate:
        run_explainer(cfg.explainer.params, trainer, model, data_module)

    if cfg.jit:
        check_jittable(model, data_module)


if __name__ == "__main__":
    my_app()
