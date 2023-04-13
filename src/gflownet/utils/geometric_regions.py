from copy import deepcopy
from typing import Tuple

import torch


class GeometricRegion(object):
    def get_sample_mask(self, points: torch.Tensor) -> torch.Tensor:
        """
        Returns a boolean mask indicating whether a point is inside a geometric region.
        """
        raise NotImplementedError

    @staticmethod
    def init_from_dict(init_dict: dict):
        """
        Builds a geometric region from a dictionary containing its kwargs.
        """
        init_dict = deepcopy(init_dict)
        region_type = init_dict.pop("type")
        if region_type == "hypersphere":
            return Hypersphere(**init_dict)
        else:
            raise ValueError(f"Unknown geometric region type {region_type}")


class Hypersphere(GeometricRegion):
    """
    Geometric region defined by a hypersphere.
    """

    def __init__(self, center: Tuple[float], radius: float):
        """
        Args:
            center: Tensor of shape (D,) containing the center of the hypersphere.
            radius: Radius of the hypersphere.
        """
        self.center = torch.tensor(center)
        self.radius = radius

    def get_sample_mask(self, points: torch.Tensor) -> torch.Tensor:
        """
        Returns a boolean mask indicating whether a point is inside the hypersphere.
        Args:
            points: Tensor of shape (N, D) containing the points to check.
        Returns:
            Tensor of shape (N,) containing the boolean mask.
        """
        return torch.norm(points - self.center, dim=1) < self.radius
