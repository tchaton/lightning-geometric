import os
import hydra

os.environ["HYDRA_FULL_ERROR"] = "1"
from omegaconf import DictConfig

from src.config import *

@hydra.main(config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    my_app()