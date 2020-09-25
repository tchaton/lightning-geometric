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
from examples.utils import attach_step_and_epoch_functions


@hydra.main(config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    data_module = instantiate(cfg.dataset)
    model = instantiate(
        cfg.model,
        optimizers=cfg.optimizers.optimizers,
        **data_module.hyper_parameters,
    )

    attach_step_and_epoch_functions(model, data_module)

    loggers = initialize_loggers(cfg, **model.config, **data_module.config)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(filepath=None, save_last=True)

    gpus = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else None

    resume_from_checkpoint = None

    trainer = instantiate(cfg.trainer, gpus=gpus, logger=loggers)

    trainer.fit(model, data_module)
    print("Training complete.")


if __name__ == "__main__":
    my_app()
