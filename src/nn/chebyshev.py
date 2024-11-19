import torch
import torch.nn as nn


class ChebyKANLayer(nn.Module):
    """
    Represents a single layer of the Chebyshev-based KAN network.
    """

    def __init__(self, input_dim, output_dim, degree):
        """
        Initialize the layer with input/output dimensions and Chebyshev polynomial degree.

        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of output features.
            degree (int): Degree of the Chebyshev polynomials used.
        """
        super(ChebyKANLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree

        # Initialize trainable coefficients for Chebyshev polynomials
        self.cheby2_coeffs = nn.Parameter(
            torch.empty(input_dim, output_dim, degree + 1)
        )
        nn.init.normal_(
            self.cheby2_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1))
        )

    def forward(self, x):
        """
        Forward pass through the layer using Chebyshev polynomials.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        # Ensure the input has the correct shape
        x = torch.reshape(x, (-1, self.inputdim))  # Reshape to (batch_size, inputdim)

        # Normalize input to [-1, 1] using Tanh (assuming x is not already normalized)
        x = torch.tanh(x)

        # Initialize tensor to store Chebyshev polynomials of the second kind
        cheby2 = torch.ones(
            x.shape[0], self.inputdim, self.degree + 1, device=x.device
        )  # Shape: (batch_size, inputdim, degree+1)

        # Compute Chebyshev polynomials using the recurrence relation
        if self.degree > 0:
            cheby2[:, :, 1] = x  # First-order polynomial
        for i in range(2, self.degree + 1):
            cheby2[:, :, i] = (
                2 * x * cheby2[:, :, i - 1].clone() - cheby2[:, :, i - 2].clone()
            )

        # Perform Chebyshev interpolation using the coefficients
        # einsum "bid,iod->bo" performs weighted summation over polynomial terms
        y = torch.einsum(
            "bid,iod->bo", cheby2, self.cheby2_coeffs
        )  # Output shape: (batch_size, output_dim)

        # Ensure the output has the correct shape
        y = y.view(-1, self.outdim)
        return y


class ChebyKAN(nn.Module):
    """
    Represents the Chebyshev-based Kolmogorov–Arnold Network (KAN).
    """

    def __init__(self, network, degree=4):
        """
        Initialize the ChebyKAN network.

        Args:
            network (list[int]): List defining the number of nodes in each layer.
            degree (int): Degree of Chebyshev polynomials for each layer.
        """
        super(ChebyKAN, self).__init__()
        self.network = network
        self.layers = nn.ModuleList()

        # Define the layers based on the specified network architecture
        for i in range(len(network) - 1):
            self.layers.append(ChebyKANLayer(network[i], network[i + 1], degree))

    def forward(self, x):
        """
        Forward pass through the entire network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        x = x.view(-1, self.network[0])  # Reshape input to match the first layer

        # Pass the input through all layers sequentially
        for _, layer in enumerate(self.layers):
            x = layer(x)
        return x


class PINNKAN(nn.Module):
    """
    Represents a Physics-Informed Neural Network (PINN) with Chebyshev-based KAN layers.
    """

    def __init__(self, network, activation):
        """
        Initialize the PINNKAN model.

        Args:
            network (list[int]): List defining the number of nodes in each layer.
            activation (nn.Module): Activation function for additional processing.
        """
        super(PINNKAN, self).__init__()
        degree = 4  # Degree of Chebyshev polynomials
        self.model = ChebyKAN(network, degree)  # Define the core ChebyKAN network
        self.activation = activation
        self.network = network
        self.layers = nn.ModuleList()

    def forward(self, x, x_min=0, x_max=1):
        """
        Forward pass for the PINNKAN model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
            x_min (float, optional): Minimum value for normalization (default: 0).
            x_max (float, optional): Maximum value for normalization (default: 1).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        return self.model(x)
