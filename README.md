# Pytorch Geometric in Pytorch Lightning

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
python train.py model={{MODEL}} dataset={{DATASET}}
```

### SUPPORTED COMBINAISONS

| `{{DATASET}}` | `{{MODEL}}` | DATASET DESCRIPTION                                                                                                                                                                       | MODEL DESCRIPTION                                                                                                                                                        | WORKING     |     |
| ------------- | ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------- | --- |
| zinc          | pna         | The ZINC dataset from the "Grammar Variational Autoencoder" <https://arxiv.org/abs/1703.01925>                                                                                            | The Principal Neighbourhood Aggregation graph convolution operator from the "Principal Neighbourhood Aggregation for Graph Nets" <https://arxiv.org/abs/2004.05718>      | True        |
| faust         | spline      | The FAUST humans dataset from the "FAUST: Dataset and Evaluation for 3D Mesh Registration" <http://files.is.tue.mpg.de/black/papers/FAUST2014.pdf>                                        | The spline-based convolutional operator from the "SplineCNN: Fast Geometric Deep Learning with Continuous B-Spline Kernels"<https://arxiv.org/abs/1711.08920>            | In progress |
| ppi           | gat         | The protein-protein interaction networks from the "Predicting Multicellular Function through Multi-layer Tissue Networks" <https://arxiv.org/abs/1707.04638>                              | The graph attentional operator from the "Graph Attention Networks" <https://arxiv.org/abs/1710.10903> True                                                               | True        |
| cora          | agnn        | The citation network datasets "Cora", "CiteSeer" and "PubMed" from the "Revisiting Semi-Supervised Learning with Graph Embeddings" <https://arxiv.org/abs/1603.08861>                     | "Attention-based Graph Neural Network for Semi-Supervised Learning" <https://arxiv.org/abs/1803.03735>                                                                   | True        |
| cora          | sage        | ""                                                                                                                                                                                        | The GraphSAGE operator from the "Inductive Representation Learning on Large Graphs" <https://arxiv.org/abs/1706.02216>                                                   | True        |
| cora          | sgc         | ""                                                                                                                                                                                        | The simple graph convolutional operator from the "Simplifying Graph Convolutional Networks" <https://arxiv.org/abs/1902.07153>                                           | True        |
| cora          | tag         | ""                                                                                                                                                                                        | The topology adaptive graph convolutional networks operator from the "Topology Adaptive Graph Convolutional Networks" <https://arxiv.org/abs/1710.10370>                 | True        |
| cora          | dna         | ""                                                                                                                                                                                        | The dynamic neighborhood aggregation operator from the "Just Jump: Towards Dynamic Neighborhood Aggregation in Graph Neural Networks" <https://arxiv.org/abs/1904.04849> | True        |
| reddit        | sage        | The Reddit dataset from the "Inductive Representation Learning on Large Graphs" <https://arxiv.org/abs/1706.02216>                                                                        | ""                                                                                                                                                                       | True        |
| reddit        | agnn        | ""                                                                                                                                                                                        | ""                                                                                                                                                                       | True        |
| icews18       | renet       | The Integrated Crisis Early Warning System (ICEWS) dataset used in the, _e.g._, "Recurrent Event Network for Reasoning over Temporal Knowledge Graphs" <https://arxiv.org/abs/1904.05530> | The Recurrent Event Network model from the "Recurrent Event Network for Reasoning over Temporal Knowledge Graphs" <https://arxiv.org/abs/1904.05530>                     | True        |
