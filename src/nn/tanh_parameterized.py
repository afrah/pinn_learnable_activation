import torch.optim
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


def xavier_initialization(layer):
    if isinstance(layer, nn.Linear):
        init.xavier_normal_(layer.weight)
        if layer.bias is not None:
            init.zeros_(layer.bias)


class TanhParametarized(nn.Module):
    def __init__(self):
        super(TanhParametarized, self).__init__()
        self.a = nn.Parameter(torch.ones(1))  # Learnable parameter for scaling
        self.b = nn.Parameter(torch.zeros(1))  # Learnable parameter for shifting

    def forward(self, x):
        return torch.tanh(self.a * x + self.b)


class PINNKAN(nn.Module):
    def __init__(self, network, activation="tanh"):
        super(PINNKAN, self).__init__()
        self.layers = nn.ModuleList()
        self.activation_function = TanhParametarized()
        self.network = network

        for index in range(len(self.network) - 1):
            self.layers.append(
                self._create_layer(self.network[index], self.network[index + 1])
            )

    def _create_layer(self, in_features, out_features):
        layer = nn.Linear(in_features, out_features)
        xavier_initialization(layer)
        return layer

    def forward(self, input, min_x=0.0, max_x=1.0):
        # input = 2.0 * (input - min_x) / (max_x - min_x) - 1.0
        for index in range(len(self.layers[:-1])):
            input = self.layers[index](input)
            input = self.activation_function(input)

        x4_output = self.layers[-1](input)
        return x4_output
