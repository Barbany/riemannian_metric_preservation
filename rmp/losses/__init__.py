import torch


class Losses:
    @staticmethod
    def get_by_name(cfg=None, name=None, *args, **kwargs):
        if cfg is not None:
            name = cfg.name
        if name == "composed":
            Loss = ComposedLoss
        elif name == "mse":
            from torch.nn import MSELoss

            return MSELoss()
        elif name == "frobenius":
            Loss = FrobeniusLoss
        elif name == "naive_projection":
            from .projection import NaiveProjectionLoss as Loss
        elif name == "normalized_projection":
            from .projection import NormalizedProjectionLoss as Loss
        elif name == "chamfer":
            from .projection import ChamferImageLoss as Loss
        else:
            raise ValueError(f"Loss {name} not recognized")
        if cfg is not None:
            cfg_copy = cfg.copy()
            del cfg_copy.name
            return Loss(*args, **cfg_copy, **kwargs)
        else:
            return Loss(*args, **kwargs)


class ComposedLoss:
    def __init__(
        self,
        projection_loss,
        time_loss,
        metric_loss,
        lambda_proj,
        lambda_time,
        lambda_metric,
        **kwargs,
    ):
        self.projection_loss = Losses.get_by_name(name=projection_loss, **kwargs)
        self.time_loss = Losses.get_by_name(name=time_loss, **kwargs)
        self.metric_loss = Losses.get_by_name(name=metric_loss, **kwargs)

        self.lambda_proj = lambda_proj
        self.lambda_time = lambda_time
        self.lambda_metric = lambda_metric

        self.need_metric = self.lambda_metric != 0

    def set(self, data_point, target_vertices=None, target_metric=None, **kwargs):
        self.projection_loss.set(data_point)
        self.target_metric = target_metric
        self.target_vertices = target_vertices

    def reset(self):
        self.projection_loss.reset()
        self.target_metric = None
        self.target_vertices = None

    def __call__(
        self,
        prediction_control=None,
        tracked_surface_prediction=None,
        riemannian_metric=None,
        **kwargs,
    ):
        intermediate_losses = {}
        if self.lambda_proj != 0:
            proj_loss_value = self.lambda_proj * self.projection_loss(
                input=tracked_surface_prediction
            )
            intermediate_losses["projection_loss"] = proj_loss_value
        if self.lambda_time != 0:
            time_loss_value = self.lambda_time * self.time_loss(
                input=prediction_control, target=self.target_vertices
            )
            intermediate_losses["time_loss"] = time_loss_value
        if self.lambda_metric != 0:
            assert riemannian_metric is not None
            metric_loss_value = self.lambda_metric * self.metric_loss(
                input=riemannian_metric, target=self.target_metric
            )
            intermediate_losses["metric_loss"] = metric_loss_value

        loss = sum(intermediate_losses.values())

        return loss, intermediate_losses


class FrobeniusLoss:
    def __init__(self, *args, **kwargs):
        pass

    @property
    def need_metric(self):
        raise NotImplementedError

    def __call__(self, input=None, target=None, *args, **kwargs):
        assert input is not None and target is not None
        return torch.linalg.vector_norm(input - target, dim=(1, 2)).pow(2).mean()

    def set(self, *args, **kwargs):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError
