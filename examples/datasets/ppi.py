import os
import os.path as osp
import numpy as np
from functools import partial
import torch
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import LightningDataModule
import torch_geometric
from torch_geometric.datasets import PPI
import torch_geometric.transforms as T

dataloader = partial(
    torch.utils.data.DataLoader, collate_fn=torch_geometric.data.batch.Batch.from_data_list, worker_init_fn=np.random.seed
)


class PPIDataset(LightningDataModule):

    NAME = 'PPI'

    def __init__(self, seed=42, num_workers=4, *args, **kwargs,):
        super().__init__(*args, **kwargs)
        self._seed = seed
        self._num_workers = num_workers

    @property
    def num_features(self):
        return 50 #TODO Find a better way to infer it

    @property
    def num_classes(self):
        return 121

    @property
    def hyper_parameters(self):
        return {"num_features": self.num_features, "num_classes": self.num_classes}   

    @staticmethod
    def _extract_index_edges(edge_index, mask):
        edge_mask = mask.nonzero().squeeze().unsqueeze(0).repeat((2, 1))
        edge_index = torch.gather(edge_index, 1, edge_mask)  
        return edge_index - edge_index.min()       

    def prepare_data(self):
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..', 'data', self.NAME)
        self.dataset_train = PPI(path, split='train')
        self.dataset_val = PPI(path, split='val')
        self.dataset_test = PPI(path, split='test')

    def train_dataloader(self, batch_size=32, transforms=None):
        loader = dataloader(
            self.dataset_train,
            batch_size=batch_size if batch_size <= len(self.dataset_train) else len(self.dataset_train),
            shuffle=True,
            num_workers=self._num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return loader

    def val_dataloader(self, batch_size=1024, transforms=None):
        loader = DataLoader(
            self.dataset_val,
            batch_size=batch_size if batch_size <= len(self.dataset_val) else len(self.dataset_val),
            shuffle=False,
            num_workers=self._num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return loader