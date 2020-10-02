from functools import partial
from hydra.utils import instantiate, get_class


class BaseTasksMixin:
    def __init__(self, *args, **kwargs):

        defaulTasksMixin = kwargs.get("defaulTasksMixin")
        assert defaulTasksMixin is not None

        mixins = [instantiate(c, *args, **kwargs) for c in defaulTasksMixin]

        named_funcs = {
            mixin: [f for f in dir(mixin) if "__" not in f] for mixin in mixins
        }

        func_names = sum(named_funcs.values(), [])
        assert len(func_names) == len(
            set(func_names)
        ), "The Tasks Mixin are overlapping. Should not be happening !"

        targets_mixin = [get_class(c._target_) for c in defaulTasksMixin]
        if len(self.__class__.__bases__) > 1:
            self.__class__.__bases__ = (self.__class__.__bases__[0],)
        for t_cls in targets_mixin:
            self.__class__.__bases__ += (t_cls,)