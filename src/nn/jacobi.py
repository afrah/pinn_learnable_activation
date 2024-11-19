import torch
import torch.nn as nn
import numpy as np


class JacobiKANLayer(nn.Module):
    """
    Represents a single layer of a Kolmogorov–Arnold Network (KAN) using Jacobi polynomials.
    """

    def __init__(self, input_dim, output_dim, degree, alpha=1, beta=1):
        """
        Initializes the Jacobi KAN layer.

        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of output features.
            degree (int): Degree of Jacobi polynomials used for approximation.
            alpha (float): Jacobi polynomial parameter (controls shape).
            beta (float): Jacobi polynomial parameter (controls shape).
        """
        super(JacobiKANLayer, self).__init__()
        self.input_dim = input_dim
        self.out_dim = output_dim
        self.degree = degree
        self.alpha = alpha
        self.beta = beta

        # Learnable coefficients for Jacobi polynomial terms
        self.jacobi_coeffs = nn.Parameter(
            torch.empty(input_dim, output_dim, degree + 1)
        )
        # Initialize coefficients using Xavier initialization
        nn.init.xavier_normal_(self.jacobi_coeffs)

    def forward(self, x):
        """
        Forward pass through the Jacobi KAN layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        # Ensure input has the correct shape
        x = torch.reshape(x, (-1, self.input_dim))
        # Normalize input to the range [-1, 1]
        x = torch.tanh(x)

        # Initialize tensor to store Jacobi polynomial values
        jacobi = torch.ones(
            x.shape[0], self.input_dim, self.degree + 1, device=x.device
        )

        # Compute Jacobi polynomials using a recurrence relation
        if self.degree > 0:
            jacobi[:, :, 1] = (
                0.5 * (self.alpha - self.beta) + (self.alpha + self.beta + 2) * x / 2
            )
        for n in range(2, self.degree + 1):
            A_n = (
                2
                * n
                * (n + self.alpha + self.beta)
                * (2 * n + self.alpha + self.beta - 2)
            )

            term1 = (
                (2 * n + self.alpha + self.beta - 1)
                * (2 * n + self.alpha + self.beta)
                * (2 * n + self.alpha + self.beta - 2)
                * x
                * jacobi[:, :, n - 1].clone()
            )
            term2 = (
                (2 * n + self.alpha + self.beta - 1)
                * (self.alpha**2 - self.beta**2)
                * jacobi[:, :, n - 1].clone()
            )
            term3 = (
                (n + self.alpha + self.beta - 1)
                * (n + self.alpha - 1)
                * (n + self.beta - 1)
                * (2 * n + self.alpha + self.beta)
                * jacobi[:, :, n - 2].clone()
            )

            jacobi[:, :, n] = (term1 - term2 - term3) / A_n

        # Combine polynomial values using learnable coefficients
        y = torch.einsum("bid,iod->bo", jacobi, self.jacobi_coeffs)
        # Reshape output to match the desired output dimensions
        y = y.view(-1, self.out_dim)
        return y


class Jacobikan(nn.Module):
    """
    Represents a multi-layer Kolmogorov–Arnold Network (KAN) using Jacobi polynomials.
    """

    def __init__(self, network, activation="", degree=4):
        """
        Initializes the Jacobikan model.

        Args:
            network (list[int]): List defining the number of nodes in each layer.
            activation (str): Placeholder for activation function (not used).
            degree (int): Degree of Jacobi polynomials for each layer.
        """
        super(Jacobikan, self).__init__()
        self.network = network

        # Define the layers based on the network architecture
        self.layers = nn.ModuleList()
        for i in range(len(network) - 1):
            self.layers.append(
                JacobiKANLayer(self.network[i], self.network[i + 1], degree)
            )

    def forward(self, x):
        """
        Forward pass through the Jacobikan model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        # Flatten the input to match the first layer's input dimensions
        x = x.view(-1, self.network[0])

        # Pass input through each layer sequentially
        for layer in self.layers:
            x = layer(x)

        return x


class PINNKAN(nn.Module):
    """
    Represents a Physics-Informed Neural Network (PINN) using JacobiKAN.
    """

    def __init__(self, network, activation):
        """
        Initializes the PINNKAN model.

        Args:
            network (list[int]): List defining the number of nodes in each layer.
            activation (nn.Module): Activation function for additional processing.
        """
        super(PINNKAN, self).__init__()
        self.model = Jacobikan(network)
        self.activation = activation
        self.network = network
        self.layers = nn.ModuleList()

    def forward(self, x, min=0, max=1):
        """
        Forward pass for the PINNKAN model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
            min (float): Minimum value for normalization (default: 0).
            max (float): Maximum value for normalization (default: 1).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        return self.model(x)
