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
        k = torch.reshape(
            torch.arange(1, self.gridsize + 1, device=x.device),
            (1, 1, 1, self.gridsize),
        )
        xrshp = torch.reshape(x, (x.shape[0], 1, x.shape[1], 1))
        c = torch.cos(k * xrshp)
        s = torch.sin(k * xrshp)
        y = torch.sum(c * self.fouriercoeffs[0:1], (-2, -1))
        y += torch.sum(s * self.fouriercoeffs[1:2], (-2, -1))
        if self.addbias:
            y += self.bias

        y = torch.reshape(y, outshape)
        return y


class PINNKAN(nn.Module):
    def __init__(self, network, activation=None, degree=4):
        super().__init__()
        self.network = network
        self.layers = nn.ModuleList()

        for i in range(len(network) - 1):
            self.layers.append(NaiveFourierKANLayer(network[i], network[i + 1], degree))

    def forward(self, x):
        x = x.view(-1, self.network[0])  

        for i, layer in enumerate(self.layers):
            x = layer(x)

        return x
