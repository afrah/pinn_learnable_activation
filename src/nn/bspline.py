# The follwoing code is a modified version of the code from the following repository:

# https://github.com/Blealtan/efficient-kan


import torch
import torch.nn.functional as F
import math
import torch.nn as nn


# KANLinear: Implements a linear layer with learnable B-spline basis functions
class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,  # Number of input features
        out_features,  # Number of output features
        grid_size=5,  # Number of grid points for B-spline basis
        spline_order=3,  # Order of the B-spline basis
        scale_noise=0.1,  # Scale for random noise initialization
        scale_base=1.0,  # Scaling factor for the base weights
        scale_spline=1.0,  # Scaling factor for spline weights
        enable_standalone_scale_spline=True,  # Whether spline scaling is standalone
        base_activation=torch.nn.SiLU,  # Activation function for the base layer
        grid_eps=0.02,  # Small offset to prevent zero division in grid calculations
        grid_range=[-1, 1],  # Range of grid for the B-splines
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Define the B-spline grid for each input feature
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer(
            "grid", grid
        )  # Register the grid as a non-trainable buffer

        # Initialize trainable weights
        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        # Scaling factors and activation function
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize the base weights using Kaiming initialization
        torch.nn.init.kaiming_normal_(
            self.base_weight, a=math.sqrt(5) * self.scale_base
        )
        with torch.no_grad():
            # Add noise to spline weights during initialization
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                torch.nn.init.kaiming_normal_(
                    self.spline_scaler, a=math.sqrt(5) * self.scale_spline
                )

    def b_splines(self, x: torch.Tensor):
        """
        Compute B-spline basis functions for input `x`.
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)  # Add dimension for grid matching
        # Initialize bases to detect which interval the input belongs to
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            # Compute B-spline basis using recurrence relations
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    @property
    def scaled_spline_weight(self):
        """
        Scale spline weights if standalone scaling is enabled.
        """
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass: Combines base output and spline output.
        """
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.view(-1, self.in_features)  # Flatten input for computation

        # Compute base layer output
        base_output = F.linear(self.base_activation(x), self.base_weight)
        # Compute spline layer output
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        # Combine base and spline outputs
        output = base_output + spline_output

        # Reshape output to match the original input shape
        output = output.view(*original_shape[:-1], self.out_features)
        return output


# KAN: Implements a network with multiple KANLinear layers
class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,  # List defining the number of nodes in each layer
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Define layers as a sequence of KANLinear modules
        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        """
        Forward pass through all layers.
        """
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute regularization loss for all layers.
        """
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )


# PINNKAN: Combines KAN with PINN
class PINNKAN(nn.Module):
    def __init__(self, network, activation):
        super(PINNKAN, self).__init__()
        self.model = KAN(network)
        self.activation = activation
        self.network = network
        self.layers = nn.ModuleList()

    def forward(self, x, x_min=0, x_max=1):
        """
        Forward pass for the PINN-KAN model.
        """
        return self.model(x)
