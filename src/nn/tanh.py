import torch.optim
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


def xavier_initialization(layer):
    if isinstance(layer, nn.Linear):
        init.xavier_normal_(layer.weight)
        if layer.bias is not None:
            init.zeros_(layer.bias)


class PINNKAN(nn.Module):
    def __init__(self, network, activation="tanh2"):
        super(PINNKAN, self).__init__()
        self.layers = nn.ModuleList()
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
        for index in range(len(self.layers[:-1])):
            input = self.layers[index](input)
            input = torch.tanh(input)

        x4_output = self.layers[-1](input)
        return x4_output
