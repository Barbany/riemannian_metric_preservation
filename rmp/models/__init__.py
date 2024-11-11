import torch.nn.functional as F


class Models:
    @staticmethod
    def get_by_name(cfg, *args, **kwargs):
        if cfg.name == "mlp":
            from .mlp import MLP as Model
        elif cfg.name == "atlasnet":
            from .atlasnet import AtlasNet as Model
        else:
            raise ValueError(f"Model {cfg.name} not recognized")

        if "device" in kwargs:
            return Model(*args, **cfg, **kwargs).to(kwargs["device"])
        else:
            return Model(*args, **cfg, **kwargs)


def get_activation(argument):
    getter = {
        "relu": F.relu,
        "sigmoid": F.sigmoid,
        "softplus": F.softplus,
        "logsigmoid": F.logsigmoid,
        "softsign": F.softsign,
        "tanh": F.tanh,
    }
    return getter.get(argument, "Invalid activation")
