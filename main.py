import torch
from torch.nn import Module, Conv2d, Linear, Dropout2d, NLLLoss
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch import cat

import numpy as np

from typing import Optional

from tqdm import tqdm

import Utils
from classical_network import ClassicalNetwork

import matplotlib.pyplot as plt

# Set train shuffle seed (for reproducibility)
torch.manual_seed(42)

#transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]) # Normalize the data for better training
transform = transforms.Compose([transforms.ToTensor()])

# Use pre-defined torchvision function to load MNIST train data
train_dataset = MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = MNIST(root="./data", train=False, download=True, transform=transform)

n_samples = None

Utils.filter_dataset(train_dataset, [0, 1], n_samples)
Utils.filter_dataset(test_dataset, [0, 1], n_samples)

# Print the target labels
print(train_dataset.targets)
print(test_dataset.targets)

# Set training parameters
epochs = 25
learning_rate = 0.001
batch_size = 10

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Create and train classical network
classical_network = ClassicalNetwork()
Utils.train_model(classical_network, train_loader, epochs, learning_rate)

# Evaluate classical network
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
accuracy = Utils.evaluate_model(classical_network, test_loader)

print(f"Accuracy of classical network: {accuracy * 100}%")

# Plot loss convergence
plt.plot(classical_network.loss_history)
plt.title("Classical NN Training Convergence")
plt.xlabel("Training Iterations")

plt.ylabel("Binary Cross Entropy Loss")
plt.show()
