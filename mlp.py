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

    #def backward(self, out, targets):
        
"""
# --- Hyper-parameters ---
num_hidden_neurons = 1
num_classes = 2
max_iters = 1000
# ------------------------

# --- Data ---
N = 4
H = 2
W = 2
X = torch.randn(N, H, W)
# ------------

# --- Data pre-preparation ---
X = X.view(N, H * W)
# ----------------------------

#print(F.one_hot(torch.arange(0, num_classes), num_classes))

model = MLP(H * W, num_hidden_neurons, num_classes)
optimizer = optim.Adam(model.parameters(), lr=3e-3)
criterion = nn.CrossEntropyLoss()

targets = torch.randint(0, num_classes, (N,))
#print(targets)

for i in range(max_iters):
    optimizer.zero_grad()
    out = model(X)
    loss = criterion(out, targets)
    loss.backward()
    optimizer.step()
    print(loss.item())
"""
