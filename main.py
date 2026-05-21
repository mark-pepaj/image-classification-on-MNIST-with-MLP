import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from mlp import MLP

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root='./data', 
    train=True, 
    download=True,
    transform=transform
)

# Load test data
test_dataset = datasets.MNIST(
    root='./data', 
    train=False, 
    transform=transform
)




H = 28
W = 28
num_classes = 10
batch_size = 128
hidden_sizes = [64, 64, 64]
max_epochs = 25
learning_rate = 1e-3

#device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = "cpu"
print(f"Using device {device}")



train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = MLP(input_size=H * W, hidden_sizes=hidden_sizes, num_output_neurons=num_classes, activations=[nn.Tanh(), nn.Tanh(), nn.Tanh()]).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()


for epoch in range(max_epochs):

    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0


    for X, targets in train_loader:

        X = X.view(X.shape[0], -1).to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, targets)
        loss.backward()
        optimizer.step()

        preds = out.argmax(dim=1)
        train_loss += loss.item()
        train_correct += (preds == targets).sum().item()
        train_total += targets.size(0) 
    
    train_loss /= len(train_loader)
    train_acc = train_correct / train_total
    train_mismatches = train_total - train_correct

    
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for X, targets in val_loader:
            X = X.view(X.shape[0], -1).to(device)
            targets = targets.to(device)

            out = model(X)
            loss = criterion(out, targets)

            preds = out.argmax(dim=1)
            val_loss += loss.item()
            val_correct += (preds == targets).sum().item()
            val_total += targets.size(0)

    val_loss /= len(val_loader)
    val_acc = val_correct / val_total
    val_mismatches = val_total - val_correct

    print("----------------------------------------------------------------------------------")
    print(f"Epoch {epoch + 1}/{max_epochs}")
    print(f"Train loss: {train_loss:.4f} -- Train acc: {train_acc:.4f} -- mismatches: {train_mismatches}")
    print(f"Val loss: {val_loss:.4f} -- Val acc: {val_acc:.4f} -- mismatches: {val_mismatches}")
