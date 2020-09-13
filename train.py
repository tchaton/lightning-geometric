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
from examples.models import *
from examples.utils import attach_step_and_epoch_functions


@hydra.main(config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    data_module = instantiate(cfg.dataset)
    model = instantiate(
        cfg.model,
        optimizer=cfg.optimizer,
        **data_module.hyper_parameters,
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(filepath=None, save_last=True)

    gpus = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else None

    resume_from_checkpoint = None

    trainer = pl.Trainer(
        max_epochs=1,
        gpus=gpus,
        limit_train_batches=1,
        limit_val_batches=4,
        limit_test_batches=4,
    )

    attach_step_and_epoch_functions(trainer, model, data_module)

    trainer.fit(model, data_module)
    print("Training complete.")


if __name__ == "__main__":
    my_app()
