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
        for t_cls in targets_mixin:
            self.__class__.__bases__ += (t_cls,)

        """
        mixins = [instantiate(c, *args, **kwargs) for c in defaulTasksMixin]

        for mixin, func_names in named_funcs.items():
            for func_name in func_names:
                func = getattr(mixin, func_name)
                partial_func = partial(func, self)
                partial_func.__code__ = func.__code__
                setattr(self, func_name, func)
        """