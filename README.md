# TorchScripted Pytorch Geometric Examples with Pytorch Lightning and Hydra

[![codecov](https://codecov.io/gh/tchaton/lightning-geometric/branch/master/graph/badge.svg)](https://codecov.io/gh/tchaton/lightning-geometric) [![Actions Status](https://github.com/tchaton/lightning-geometric/workflows/unittest/badge.svg)](https://github.com/tchaton/lightning-geometric/actions)

### Setup on MacOs. Please, adapt to others OS :)

```python
brew install cmake
pyenv install 3.7.8
pyenv local 3.7.8
python -m venv
source .venv/bin/activate
poetry install
```

### PRINCIPAL CMD

```python
python train.py model={{MODEL}} dataset={{DATASET}} loggers={{LOGGERS}} log={{LOG}} notes={{NOTES}} name={{NAME}} jit={{JIT}}
```

- `LOGGERS` str: Configuration file to log to Wandb, currently using mine as `thomas-chaton`
- `LOG` bool: Wheter to log training to wandb
- `NOTES` str: A note associated to the training
- `NAME` str: Training name appearing on Wandb.
- `JIT` bool: Wheter to make model jittable.

### Working Inference

Have a look at `test/test_inference.py`

### SUPPORTED COMBINAISONS

| `{{DATASET}}` | `{{MODEL}}` | DATASET DESCRIPTION                                                                                                                                                                       | MODEL DESCRIPTION                                                                                                                                                                          | WORKING                      |     |
| ------------- | ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------- | --- |
| zinc          | pna         | The ZINC dataset from the "Grammar Variational Autoencoder" <https://arxiv.org/abs/1703.01925>                                                                                            | The Principal Neighbourhood Aggregation graph convolution operator from the "Principal Neighbourhood Aggregation for Graph Nets" <https://arxiv.org/abs/2004.05718>                        | True                         |
| faust         | spline      | The FAUST humans dataset from the "FAUST: Dataset and Evaluation for 3D Mesh Registration" <http://files.is.tue.mpg.de/black/papers/FAUST2014.pdf>                                        | The spline-based convolutional operator from the "SplineCNN: Fast Geometric Deep Learning with Continuous B-Spline Kernels"<https://arxiv.org/abs/1711.08920>                              | In progress                  |
| ppi           | gat         | The protein-protein interaction networks from the "Predicting Multicellular Function through Multi-layer Tissue Networks" <https://arxiv.org/abs/1707.04638>                              | The graph attentional operator from the "Graph Attention Networks" <https://arxiv.org/abs/1710.10903> True                                                                                 | True                         |
| cora          | agnn        | The citation network datasets "Cora", "CiteSeer" and "PubMed" from the "Revisiting Semi-Supervised Learning with Graph Embeddings" <https://arxiv.org/abs/1603.08861>                     | "Attention-based Graph Neural Network for Semi-Supervised Learning" <https://arxiv.org/abs/1803.03735>                                                                                     | True                         |
| cora          | sage        | ""                                                                                                                                                                                        | The GraphSAGE operator from the "Inductive Representation Learning on Large Graphs" <https://arxiv.org/abs/1706.02216>                                                                     | True                         |
| cora          | sgc         | ""                                                                                                                                                                                        | The simple graph convolutional operator from the "Simplifying Graph Convolutional Networks" <https://arxiv.org/abs/1902.07153>                                                             | True                         |
| cora          | tag         | ""                                                                                                                                                                                        | The topology adaptive graph convolutional networks operator from the "Topology Adaptive Graph Convolutional Networks" <https://arxiv.org/abs/1710.10370>                                   | True                         |
| cora          | dna         | ""                                                                                                                                                                                        | The dynamic neighborhood aggregation operator from the "Just Jump: Towards Dynamic Neighborhood Aggregation in Graph Neural Networks" <https://arxiv.org/abs/1904.04849>                   | True                         |
| reddit        | sage        | The Reddit dataset from the "Inductive Representation Learning on Large Graphs" <https://arxiv.org/abs/1706.02216>                                                                        | ""                                                                                                                                                                                         | True                         |
| reddit        | agnn        | ""                                                                                                                                                                                        | ""                                                                                                                                                                                         | True                         |
| icews18       | renet       | The Integrated Crisis Early Warning System (ICEWS) dataset used in the, _e.g._, "Recurrent Event Network for Reasoning over Temporal Knowledge Graphs" <https://arxiv.org/abs/1904.05530> | The Recurrent Event Network model from the "Recurrent Event Network for Reasoning over Temporal Knowledge Graphs" <https://arxiv.org/abs/1904.05530>                                       | Waiting for support for TGCN |
| cora          | argva       | ""                                                                                                                                                                                        | The Adversarially Regularized Variational Graph Auto-Encoder model from the "Adversarially Regularized Graph Autoencoder for Graph Embedding" <https://arxiv.org/abs/1802.04407>`          | True                         |
| cora          | arma        | ""                                                                                                                                                                                        | The ARMA graph convolutional operator from the "Graph Neural Networks with Convolutional ARMA Filters" <https://arxiv.org/abs/1901>.01343>                                                 | True                         |
| cora          | gcn         | ""                                                                                                                                                                                        | The GCN graph convolutional operator from the "Semi Supervised Classification with Graph Convolution Networks" <https://arxiv.org/pdf/1609.02907.pdf>.01343>                               | True                         |
| cora          | gcn2        | ""                                                                                                                                                                                        | The graph convolutional operator with initial residual connections and identity mapping (GCNII) from the "Simple and Deep Graph Convolutional Networks" <https://arxiv.org/abs/2007.02133> | True                         |

# DATASET SIZES

```
529M    ./Flickr
 74M    ./FAUST
 16M    ./cora
3.5G    ./Reddit
383M    ./ZINC
1.8G    ./MNISTSuperpixels
182M    ./OgbnArxiv
192M    ./PPI
156M    ./ICEWS18
6.8G    .
```
