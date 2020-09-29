import os.path as osp

import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import (
    v_measure_score,
    homogeneity_score,
    completeness_score,
)
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv, ARGVA
from examples.core.base_model import BaseModel


class Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Encoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=True)
        self.conv_mu = GCNConv(hidden_channels, out_channels, cached=True)
        self.conv_logstd = GCNConv(hidden_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


class Discriminator(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Discriminator, self).__init__()
        self.lin1 = torch.nn.Linear(in_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin3 = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x


class ARGVANodeClustering(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.save_hyperparameters()

        num_features = kwargs["num_features"]
        hidden_channels = kwargs["hidden_channels"]
        in_channels = kwargs["in_channels"]
        out_channels = kwargs["out_channels"]

        self._n_critic = kwargs["n_critic"]

        self.encoder = Encoder(
            num_features, hidden_channels=hidden_channels, out_channels=out_channels
        )

        self.discriminator = Discriminator(
            in_channels=in_channels,
            hidden_channels=2 * hidden_channels,
            out_channels=in_channels,
        )

        self.model = ARGVA(self.encoder, self.discriminator)

    def configure_optimizers(self):
        optim_encoder = self._init_optim("encoder", self.encoder.parameters())
        optim_discriminator = self._init_optim(
            "discriminator", self.discriminator.parameters()
        )
        return (
            {"optimizer": optim_discriminator, "frequency": self._n_critic},
            {"optimizer": optim_encoder, "frequency": 1},
        )

    def forward(self, batch):
        x = batch.x
        edge_index = batch.edge_index[0]
        z = self.model.encode(x, edge_index)
        loss = self.model.recon_loss(z, edge_index)
        loss = loss + (1 / x.shape[0]) * self.model.kl_loss()
        return z, loss
