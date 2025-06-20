import torch
import torch.nn as nn


class ChebyKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(ChebyKANLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree

        self.cheby2_coeffs = nn.Parameter(
            torch.empty(input_dim, output_dim, degree + 1)
        )
        nn.init.normal_(
            self.cheby2_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1))
        )

    def forward(self, x):
        x = torch.reshape(x, (-1, self.inputdim))  # shape = (batch_size, inputdim)
        # Since Chebyshev polynomial is defined in [-1, 1]
        # We need to normalize x to [-1, 1] using tanh
        x = torch.tanh(x)  # Normalize x to [-1, 1]

        cheby2 = torch.ones(x.shape[0], self.inputdim, self.degree + 1, device=x.device)
        if self.degree > 0:
            cheby2[:, :, 1] = x
        for i in range(2, self.degree + 1):
            cheby2[:, :, i] = (
                2 * x * cheby2[:, :, i - 1].clone() - cheby2[:, :, i - 2].clone()
            )
        y = torch.einsum(
            "bid,iod->bo", cheby2, self.cheby2_coeffs
        )  # shape = (batch_size, outdim)
        y = y.view(-1, self.outdim)
        return y


class ChebyKAN(nn.Module):
    def __init__(self, network, degree=4):
        super(ChebyKAN, self).__init__()
        self.network = network
        self.layers = nn.ModuleList()

        for i in range(len(network) - 1):
            self.layers.append(ChebyKANLayer(network[i], network[i + 1], degree))

            # if i < len(network) - 2:
            #     self.layer_norms.append(nn.LayerNorm(network[i + 1]))

    def forward(self, x):
        x = x.view(-1, self.network[0])  # Flatten the images

        for i, layer in enumerate(self.layers):
            x = layer(x)
            # # Apply LayerNorm if it's not the last layer
            # if i < len(self.layer_norms):
            #     x = self.layer_norms[i](x)

        return x


class PINNKAN(nn.Module):
    def __init__(self, network, activation=None):
        super(PINNKAN, self).__init__()
        degree = 4
        self.model = ChebyKAN(network, degree)
        self.activation = activation
        self.network = network
        self.layers = nn.ModuleList()

    def forward(self, x, x_min=0, x_max=1):
        return self.model(x)
