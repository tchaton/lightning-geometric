from hydra.utils import instantiate, get_class
from omegaconf import DictConfig
from enum import Enum
from functools import partial
from torch_geometric.data import Batch
from torch_geometric.data import DataLoader
from torch_geometric.data import NeighborSampler


class SAMPLING(Enum):
    DataLoader = "dataloader"
    NeighborSampler = "neighbor_sampler"
    GraphSAINTRandomWalkSampler = "graph_saint_random_walk_sampler"


def find_enum(sampling_str, en):
    for e in en:
        if sampling_str == e.name:
            return e
    raise Exception(
        f"Provided name {sampling_str} wasn't found in {[e.name for e in en]}"
    )


class BaseDatasetSamplerMixin:

    STAGES = ["train", "val", "test"]

    def __init__(self, *args, **kwargs):

        samplers = None
        self._samplers = kwargs.get("samplers")
        if self._samplers:
            for stage_sampler in self._samplers:
                stage = stage_sampler.stage
                func_name = f"{stage}_dataloader"
                sampling = stage_sampler.sampling
                if sampling == SAMPLING.DataLoader.value:
                    func = partial(self.create_dataloader, stage=stage)
                    func.__code__ = self.create_dataloader.__code__
                    setattr(self, f"{stage}_loader_type", SAMPLING.DataLoader.value)
                elif sampling == SAMPLING.NeighborSampler.value:
                    func = partial(self.create_neighbor_sampler, stage=stage)
                    func.__code__ = self.create_neighbor_sampler.__code__
                    setattr(
                        self, f"{stage}_loader_type", SAMPLING.NeighborSampler.value
                    )
                else:
                    if hasattr(sampling, "_target_"):
                        samplers = [
                            DictConfig(
                                {
                                    "stage": s["stage"],
                                    "sampling": find_enum(
                                        s["sampling"].name, SAMPLING
                                    ).value,
                                }
                            )
                            for s in self._samplers
                        ]
                        loader_cls = get_class(sampling._target_)
                        params = {}
                        if hasattr(sampling, "params"):
                            params = sampling.params
                        func = partial(
                            self.create_loader_from_cls,
                            loader_cls=loader_cls,
                            params=params,
                            stage=stage,
                        )
                        func.__code__ = self.create_loader_from_cls.__code__
                    else:
                        raise Exception(
                            f"Strategy should be within {[v.value for v in SAMPLING]}"
                        )
                setattr(self, func_name, func)

        if samplers is not None:
            self._samplers = samplers

        self._num_edges = kwargs.get("num_edges")
        self._num_layers = kwargs.get("num_layers")
        if (self._num_edges is not None) and (self._num_layers is not None):
            self._sizes = [self._num_edges, self._num_layers]

    def create_dataloader(self, batch_size=2, transforms=None, stage=None):
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

    def create_neighbor_sampler(self, batch_size=2, transforms=None, stage=None):
        return NeighborSampler(
            self.data.edge_index,
            node_idx=getattr(self.data, f"{stage}_mask"),
            sizes=self._sizes,
            num_workers=self._num_workers,
            drop_last=self._drop_last,
            pin_memory=self._pin_memory,
        )

    def create_loader_from_cls(self, loader_cls=None, params=None, stage=None):
        if not hasattr(self, "_loaded_dataset"):
            self._loaded_dataset = loader_cls(self.data, **params)
        dataset = Batch.from_data_list([d for d in self._loaded_dataset])
        return NeighborSampler(
            dataset.edge_index,
            node_idx=getattr(dataset, f"{stage}_mask"),
            sizes=self._sizes,
            num_workers=self._num_workers,
            drop_last=self._drop_last,
            pin_memory=self._pin_memory,
        )
