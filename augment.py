import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt


random_rotate = transforms.RandomRotation(10) # 10 degrees
random_affine = transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10)
horizontal_flip = transforms.RandomHorizontalFlip(p=0.5) # prob that it flips
vertical_flip = transforms.RandomVerticalFlip(p=0.5) # prob that it flips
augment_shape = transforms.RandomResizedCrop((28, 28), scale=(0.5, 1), ratio=(0.5, 2))
augment_color = transforms.ColorJitter(brightness=0.5, contrast=0, saturation=0, hue=0)

train_augments = transforms.Compose([
    random_affine,
    transforms.ToTensor()
    ])
test_augments = transforms.Compose([transforms.ToTensor()])


train_dataset = datasets.MNIST(
    root='./data', 
    train=True, 
    download=True,
    transform=train_augments
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

for i, (X, target) in enumerate(train_loader):
    X = X.view(-1, X.shape[-2], X.shape[-1])
    for j in range(X.shape[0]):
        plt.imshow(X[j])
        plt.show()
    break
