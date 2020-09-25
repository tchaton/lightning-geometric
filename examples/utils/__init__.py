from pytorch_lightning.utilities.model_utils import is_overridden


def attach_step_and_epoch_functions(model, datamodule):
    datamodule.forward = model.forward
    for attr in dir(datamodule):
        if sum([token in attr for token in ["_step", "_epoch_end"]]) > 0:
            if not is_overridden(attr, model):
                setattr(model, attr, getattr(datamodule, attr))