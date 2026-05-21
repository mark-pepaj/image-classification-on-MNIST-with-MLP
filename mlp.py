import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_output_neurons):
        super().__init__()

        layer_sizes = [input_size] + hidden_sizes

        self.hidden_layers = nn.ModuleList([
            nn.Linear(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(hidden_sizes))
        ])

        self.output_layer = nn.Linear(hidden_sizes[-1], num_output_neurons)
        self.tanh = nn.Tanh()

    def forward(self, x):
        for layer in self.hidden_layers:
            x = self.tanh(layer(x))

        return self.output_layer(x)
