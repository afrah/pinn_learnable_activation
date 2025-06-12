import numpy as np
import torch
import torch.nn as nn


class NaiveFourierKANLayer(torch.nn.Module):
    def __init__(
        self, inputdim, outdim, gridsize, addbias=True, smooth_initialization=False
    ):
        super(NaiveFourierKANLayer, self).__init__()
        self.gridsize = gridsize
        self.addbias = addbias
        self.inputdim = inputdim
        self.outdim = outdim

        grid_norm_factor = (
            (torch.arange(gridsize) + 1) ** 2
            if smooth_initialization
            else np.sqrt(gridsize)
        )

        self.fouriercoeffs = torch.nn.Parameter(
            torch.randn(2, outdim, inputdim, gridsize)
            / (np.sqrt(inputdim) * grid_norm_factor)
        )
        if self.addbias:
            self.bias = torch.nn.Parameter(torch.zeros(1, outdim))

    # x.shape ( ... , indim )
    # out.shape ( ..., outdim)
    def forward(self, x):
        xshp = x.shape
        outshape = xshp[0:-1] + (self.outdim,)
        x = torch.reshape(x, (-1, self.inputdim))
        # Starting at 1 because constant terms are in the bias
        k = torch.reshape(
            torch.arange(1, self.gridsize + 1, device=x.device),
            (1, 1, 1, self.gridsize),
        )
        xrshp = torch.reshape(x, (x.shape[0], 1, x.shape[1], 1))
        # This should be fused to avoid materializing memory
        c = torch.cos(k * xrshp)
        s = torch.sin(k * xrshp)
        # We compute the interpolation of the various functions defined by
        # their fourier coefficient for each input coordinates and we sum them
        y = torch.sum(c * self.fouriercoeffs[0:1], (-2, -1))
        y += torch.sum(s * self.fouriercoeffs[1:2], (-2, -1))
        if self.addbias:
            y += self.bias
        # End fuse

        # You can use einsum instead to reduce memory usage
        # It stills not as good as fully fused but it should help
        # einsum is usually slower though
        # c2 = torch.reshape(c, (1, x.shape[0], x.shape[1], self.gridsize))
        # s2 = torch.reshape(s, (1, x.shape[0], x.shape[1], self.gridsize))
        # y2 = torch.einsum(
        #     "dbik,djik->bj", torch.concat([c2, s2], axis=0), self.fouriercoeffs
        # )
        # if self.addbias:
        #     y2 += self.bias
        # diff = torch.sum((y2 - y) ** 2)
        # print(f"{diff.item()=}")
        # print(diff)  # should be ~0

        y = torch.reshape(y, outshape)
        return y


class PINNKAN(nn.Module):
    def __init__(self, network, activation=None, degree=4):
        super().__init__()
        self.network = network
        self.layers = nn.ModuleList()
        # self.layer_norms = nn.ModuleList()

        for i in range(len(network) - 1):
            self.layers.append(NaiveFourierKANLayer(network[i], network[i + 1], degree))

    def forward(self, x):
        x = x.view(-1, self.network[0])  # Flatten the images

        # Forward pass through layers using a loop
        for i, layer in enumerate(self.layers):
            x = layer(x)

        return x
