import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_output_neurons, activations=None, output_activation=None):
        super().__init__()

        if activations is None:
            activations = [nn.Tanh() for _ in hidden_sizes]
        elif isinstance(activations, nn.Module):
            activations = [activations] * len(hidden_sizes)

        assert len(activations) == len(hidden_sizes), (
            f"activations length ({len(activations)}) must match hidden_sizes length ({len(hidden_sizes)})"
        )

        layer_sizes = [input_size] + hidden_sizes

        self.hidden_layers = nn.ModuleList([
            nn.Linear(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(hidden_sizes))
        ])
        self.activations = nn.ModuleList(activations)
        self.output_layer = nn.Linear(hidden_sizes[-1], num_output_neurons)
        self.output_activation = output_activation

    def forward(self, x):
        for layer, act in zip(self.hidden_layers, self.activations):
            x = act(layer(x))

        x = self.output_layer(x)

        if self.output_activation is not None:
            x = self.output_activation(x)

        return x
