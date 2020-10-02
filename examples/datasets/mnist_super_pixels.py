import os
import os.path as osp
from torch_geometric.datasets import  MNISTSuperpixels
from examples.core.base_dataset import BaseDataset
from examples.core.transforms import AddFeatsByKeys


class MNISTSuperpixelsDataset(BaseDataset):

    NAME = "MNISTSuperpixels"

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self._num_features = 1
        if isinstance(self._transform.transforms[-1], AddFeatsByKeys):
            if sum(self._transform.transforms[-1]._list_add_to_x) == 1:
                self._num_features += 2

    @property
    def num_features(self):
        return self._num_features

    @property
    def num_classes(self):
        return 10

    @property
    def edge_attr(self):
        return 2

    def prepare_data(self):
        path = osp.join(
            osp.dirname(osp.realpath(__file__)), "..", "..", "data", self.NAME
        )
        self.train_dataset = MNISTSuperpixels(path, True, transform=self._transform)
        self.test_dataset = MNISTSuperpixels(path, False, transform=self._transform)
