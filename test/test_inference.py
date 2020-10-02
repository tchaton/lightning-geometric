import os
import sys
import warnings
warnings.simplefilter("ignore")
from omegaconf import OmegaConf
import pytest
sys.path.append('..')
from hydra.experimental import compose, initialize

from examples.config import cs
from train import train

DIR_PATH = os.path.dirname(os.path.dirname(__file__))
def getcwd():
    return os.path.join(DIR_PATH, "outputs")

os.getcwd = getcwd

@pytest.mark.parametrize("model", ["argva"])
@pytest.mark.parametrize("dataset", ["cora"])
@pytest.mark.parametrize("jit", ["False"])
def test_argva_cora_inference(model, dataset, jit):
    cmd_line = "model={} dataset={} loggers=thomas-chaton optimizers=vgae log=false notes='' name=test jit={} explain=False"
    with initialize(config_path="../conf", job_name="test_app"):
        print({"model":model, "dataset":dataset, "jit":jit})
        cfg = compose(config_name="config", overrides=cmd_line.format(model, dataset, jit).split(' '))
        train(cfg)

@pytest.mark.parametrize("model", ["sage", "dna", "dna", "gcn", "sgc", "tag"])
@pytest.mark.parametrize("dataset", ["cora"])
@pytest.mark.parametrize("jit", ["True", "False"])
def test_cora_inference(model, dataset, jit):
    cmd_line = "model={} dataset={} loggers=thomas-chaton log=False notes='' name='test' explain=False jit={}"
    with initialize(config_path="../conf", job_name="test_app"):
        print({"model":model, "dataset":dataset, "jit":jit})
        cfg = compose(config_name="config", overrides=cmd_line.format(model, dataset, jit).split(' '))
        train(cfg)
