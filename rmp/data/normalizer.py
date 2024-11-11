import numpy as np
import torch


class PCNormalizer:
    def __init__(self, normalize_pc, dataset_statistics, device):
        self.normalize_pc = normalize_pc
        self.dataset_statistics = dataset_statistics
        self.torch_statistics = {
            k: (torch.from_numpy(v).float().to(device) if isinstance(v, np.ndarray) else v)
            for k, v in dataset_statistics.items()
        }

    def __call__(self, points):
        if self.normalize_pc:
            if torch.is_tensor(points):
                return (points - self.torch_statistics["mean"]) / self.torch_statistics["max_norm"]
            else:
                return (points - self.dataset_statistics["mean"]) / self.dataset_statistics[
                    "max_norm"
                ]
        else:
            return points

    def camera_frame(self, points):
        if self.normalize_pc:
            if torch.is_tensor(points):
                return points + self.torch_statistics["mean"] / self.torch_statistics["max_norm"]
            else:
                return (
                    points + self.dataset_statistics["mean"] / self.dataset_statistics["max_norm"]
                )
        else:
            return points

    def inverse(self, points):
        if self.normalize_pc:
            if torch.is_tensor(points):
                return (points * self.torch_statistics["max_norm"]) + self.torch_statistics["mean"]
            else:
                return (points * self.dataset_statistics["max_norm"]) + self.dataset_statistics[
                    "mean"
                ]
        else:
            return points
