import torch
from torch.nn import Module, Conv2d, Linear, Dropout2d, NLLLoss
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch import cat
from torchsummary import summary
from torchviz import make_dot

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

# Set training parameters
epochs = 25
learning_rate = 0.001
batch_size = 10
n_samples = 50

# Create data loaders
#train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
train_loader, test_loader = Utils.get_data_loaders_from_labels([0, 1], n_samples, batch_size, transform)

# Create and train classical network
classical_network = ClassicalNetwork()
summary(classical_network, (1, 28, 28))
Utils.visualise_model_architecture(classical_network)


Utils.train_model(classical_network, train_loader, epochs, learning_rate)

# Evaluate classical network
accuracy = Utils.evaluate_model(classical_network, test_loader)

print(f"Accuracy of classical network: {accuracy * 100}%")

# Plot loss convergence
Utils.visualise_loss_history(classical_network)
