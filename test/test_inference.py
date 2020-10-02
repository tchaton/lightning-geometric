import os
import sys
import pytest
sys.path.append('..')
from hydra.experimental import compose, initialize

#from examples.config import cs

cmd_line = "model={} dataset={} loggers=thomas-chaton log=False notes='' name='test' jit={}"

model = ["agnn", "argva", "arma", "dna", "gcn", "gcn2", "infomax", "pna", "sgc", "tag"]
dataset = ["cora"]
jit = ["False", "True"]

@pytest.mark.parametrize("model", model)
@pytest.mark.parametrize("dataset", dataset)
@pytest.mark.parametrize("jit", jit)

def test_inference(model, dataset, jit):
    pass
    #with initialize(config_path="../conf", job_name="test_app"):
    #    cfg = compose(config_name="config", overrides=cmd_line.format(model, dataset, jit).split(' '))
    #    print(OmegaConf.to_yaml(cfg))