from enum import Enum
from functools import partial
from torch_geometric.data import DataLoader
from torch_geometric.data import NeighborSampler


class SAMPLING(Enum):
    DataLoader = "dataloader"
    NeighborSampler = "neighbor_sampler"


class BaseDatasetSampler:

    STAGES = ["train", "val", "test"]

    def __init__(self, *args, **kwargs):

        self._samplers = kwargs.get("samplers")
        if self._samplers:
            for stage_sampler in self._samplers:
                stage = stage_sampler.stage
                func_name = f"{stage}_dataloader"
                sampling = stage_sampler.sampling
                if sampling == SAMPLING.DataLoader.value:
                    func = partial(self.create_dataloader, stage=stage)
                    func.__code__ = self.create_dataloader.__code__
                elif sampling == SAMPLING.NeighborSampler.value:
                    func = partial(self.create_neighbor_sampler, stage=stage)
                    func.__code__ = self.create_neighbor_sampler.__code__
                else:
                    raise Exception(
                        f"Strategy should be within {[v.value for v in SAMPLING]}"
                    )
                setattr(self, func_name, func)

        self._num_edges = kwargs.get("num_edges")
        self._num_layers = kwargs.get("num_layers")
        if (self._num_edges is not None) and (self._num_layers is not None):
            self._sizes = [self._num_edges, self._num_layers]

    def create_dataloader(self, batch_size=1, transforms=None, stage=None):
        try:
            dataset = getattr(self, f"{stage}_dataset")
        except:
            dataset = getattr(self, "dataset")
        return DataLoader(
            dataset,
            batch_size=batch_size if batch_size <= len(dataset) else len(dataset),
            shuffle=self._shuffle,
            num_workers=self._num_workers,
            drop_last=self._drop_last,
            pin_memory=self._pin_memory,
            follow_batch=self._follow_batch,
        )

    def create_neighbor_sampler(self, batch_size=1, transforms=None, stage=None):
        return NeighborSampler(
            self.data.edge_index,
            node_idx=getattr(self.data, f"{stage}_mask"),
            sizes=self._sizes,
            num_workers=self._num_workers,
            drop_last=self._drop_last,
            pin_memory=self._pin_memory,
        )