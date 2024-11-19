# The follwoing code is a modified version of the code from the following repository:

# https://github.com/ZiyaoLi/fast-kan/blob/master/setup.py


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import *


class SplineLinear(nn.Linear):
    """
    A linear layer specifically for spline-based weight updates, initialized with
    truncated normal distribution for stability.
    """

    def __init__(
        self, in_features: int, out_features: int, init_scale: float = 0.1, **kw
    ):
        self.init_scale = init_scale
        super().__init__(in_features, out_features, bias=False, **kw)

    def reset_parameters(self) -> None:
        # Truncated normal initialization for weights
        nn.init.trunc_normal_(self.weight, mean=0, std=self.init_scale)


class RadialBasisFunction(nn.Module):
    """
    Computes Radial Basis Functions (Gaussian) for given input and grid points.
    """

    def __init__(
        self,
        grid_min: float = -2.0,  # Minimum value of the RBF grid
        grid_max: float = 2.0,  # Maximum value of the RBF grid
        num_grids: int = 8,  # Number of grid points
        denominator: float = None,  # Smoothing parameter (controls RBF spread)
    ):
        super().__init__()
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids
        # Define grid points
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = torch.nn.Parameter(grid, requires_grad=False)
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)

    def forward(self, x):
        # Compute RBF values using Gaussian kernel
        return torch.exp(-(((x[..., None] - self.grid) / self.denominator) ** 2))


class FastKANLayer(nn.Module):
    """
    Implements a single layer of the FastKAN framework, combining RBF transformations
    and spline-based weight updates.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        num_grids: int = 8,
        use_base_update: bool = True,
        use_layernorm: bool = False,
        base_activation=F.silu,
        spline_weight_init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layernorm = None

        # Layer normalization (optional)
        if use_layernorm:
            assert input_dim > 1, "Do not use layernorms on 1D inputs."
            self.layernorm = nn.LayerNorm(input_dim)

        # RBF transformation
        self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)

        # Linear transformation for RBF basis
        self.spline_linear = SplineLinear(
            input_dim * num_grids, output_dim, spline_weight_init_scale
        )

        self.use_base_update = use_base_update
        if use_base_update:
            self.base_activation = base_activation
            self.base_linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, use_layernorm=True):
        # Apply layer normalization if enabled
        if self.layernorm is not None and use_layernorm:
            spline_basis = self.rbf(self.layernorm(x))
        else:
            spline_basis = self.rbf(x)

        # Compute spline-based output
        ret = self.spline_linear(spline_basis.view(*spline_basis.shape[:-2], -1))

        # Optionally add base update
        if self.use_base_update:
            base = self.base_linear(self.base_activation(x))
            ret = ret + base
        return ret

    def plot_curve(
        self, input_index, output_index, num_pts=1000, num_extrapolate_bins=2
    ):
        """
        Visualize the learned curves for a specific input-output pair.
        """
        # Ensure the indices are within valid ranges
        assert input_index < self.input_dim
        assert output_index < self.output_dim

        ng = self.rbf.num_grids  # Number of grid points
        h = self.rbf.denominator  # Grid spacing
        w = self.spline_linear.weight[
            output_index, input_index * ng : (input_index + 1) * ng
        ]  # Extract spline weights
        x = torch.linspace(
            self.rbf.grid_min - num_extrapolate_bins * h,
            self.rbf.grid_max + num_extrapolate_bins * h,
            num_pts,
        )  # Define the range for visualization
        with torch.no_grad():
            y = (w * self.rbf(x.to(w.dtype))).sum(-1)  # Compute the curve
        return x, y
