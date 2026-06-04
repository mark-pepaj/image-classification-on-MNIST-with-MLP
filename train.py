import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from multi_layer_perceptron.model import *
from data_augmentation_for_deep_learning.augment import *



augmentor = Augmentor(
        pad=2,
        augment_configs=[RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),]
        )


train_dataset = datasets.MNIST(
    root='./data', 
    train=True, 
    download=True,
    transform=augmentor.get_train_transforms()
)

# Load test data
val_dataset = datasets.MNIST(
    root='./data', 
    train=False, 
    transform=augmentor.get_val_transforms()
)


num_epochs = 100
batch_size = 32
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using Device: {device}")


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

layer_configs = [
        Flatten(),
        LinearLayer(in_features=32*32, out_features=32),
        Activation(name="Tanh"),
        LinearLayer(in_features=32, out_features=64),
        Activation(name="Tanh"),
        LinearLayer(in_features=64, out_features=32),
        Activation(name="Tanh"),
        LinearLayer(in_features=32, out_features=10),
]

model = MLP(layer_configs=layer_configs).to(device)
criterion_config = Criterion(name="CrossEntropy")
optimizer_config = Optimizer(name="AdamW", lr=1e-3, weight_decay=0.01)


history, best_val_loss, best_val_epoch = model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        layer_configs=layer_configs,
        criterion_config=criterion_config,
        optimizer_config=optimizer_config,
        )


print(f"Best Validation Loss: {best_val_loss}, Epoch: {best_val_epoch}")

plt.plot(history["train_loss"], label="train loss")
plt.plot(history["val_loss"], label="val loss")
plt.title("Training/Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

plt.plot(history["val_acc"], label="val acc")
plt.title("Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()
