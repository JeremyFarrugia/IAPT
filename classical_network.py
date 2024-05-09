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



def filter_dataset(dataset: MNIST, labels: list, n_samples: Optional[int]) -> None:
    """
    Filter to only include specified labels and number of samples
    labels: The labels to include in the dataset
    n_samples: The number of samples to include for each label
    If n_samples is None, all samples of the specified labels will be included
    """

    idx = np.array([])
    if n_samples is None:
        idx = np.where(np.isin(dataset.targets, labels))[0]
    else:
        for label in labels:
            idx = np.append(idx, np.where(dataset.targets == label)[0][:n_samples])
    dataset.data = dataset.data[idx]
    dataset.targets = dataset.targets[idx]


class ClassicalNetwork(Module):
    """
    Classical neural network for MNIST classification
    Used as a comparison to the quantum neural network
    Uses two convolutional layers and two fully connected layers
    """

    def __init__(self):
        super(ClassicalNetwork, self).__init__()
        self.conv1 = Conv2d(1, 2, kernel_size=5)
        self.conv2 = Conv2d(2, 16, kernel_size=5)
        self.dropout = Dropout2d()
        self.fc1 = Linear(256, 64)
        self.fc2 = Linear(64, 2)
        self.fc3 = Linear(2, 1)

        self.loss_function = torch.nn.BCELoss()

        self.loss_history = []
        self.trained = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, kernel_size=2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, kernel_size=2)
        x = self.dropout(x)
        x = x.view(x.shape[0], -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x
    
    def train_model(self, train_loader: DataLoader, epochs: int, learning_rate: float):
        """
        Train the network using the specified data loader, number of epochs, and learning rate
        Optimiser: Adam
        Loss function: Negative Log Likelihood Loss
        """

        if self.trained:
            print("Model has already been trained. To retrain, create a new model.")
            return

        self.loss_history = []
        optimiser = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = self.loss_function
        self.train() # Set model to training mode

        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(tqdm(train_loader,desc=f"Epoch {epoch + 1}")):
                inputs, labels = data
                optimiser.zero_grad()
                outputs = self(inputs)
                labels = labels.unsqueeze(1).float()
                loss = criterion(outputs, labels)
                loss.backward()
                optimiser.step()
                running_loss += loss.item()
            print(f"Epoch {epoch + 1}, loss: {running_loss / len(train_loader)}")
            self.loss_history.append(running_loss / len(train_loader))
            
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict the output of the network for the given input
        """
        self.eval()
        with torch.no_grad():
            return self(x)
            

    def evaluate_model(self, test_loader: DataLoader) -> float:
        """
        Evaluate the model using the specified data loader
        Returns the accuracy of the model
        """

        loss_func = self.loss_function

        self.eval()
        correct = 0
        total = 0

        total_loss = []

        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                outputs = self(inputs)
                if len(outputs.shape) == 1:
                    outputs = outputs.reshape(1, *outputs.shape)

                print(outputs)
                print("----")
                print(labels)

                pred = outputs.round()
                correct += pred.eq(labels.view_as(pred)).sum().item()
                total += labels.size(0)

                outputs = outputs.squeeze(1).float()
                loss = loss_func(outputs, labels.float())
                total_loss.append(loss.item())

            print(f"Test loss: {sum(total_loss) / len(total_loss)}")
            print(f"Accuracy: {correct / total}")
            return correct / total

# Set train shuffle seed (for reproducibility)
torch.manual_seed(42)

#transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]) # Normalize the data for better training
transform = transforms.Compose([transforms.ToTensor()])

# Use pre-defined torchvision function to load MNIST train data
train_dataset = MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = MNIST(root="./data", train=False, download=True, transform=transform)

n_samples = 250

filter_dataset(train_dataset, [0, 1], n_samples)
filter_dataset(test_dataset, [0, 1], n_samples)

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
classical_network.train_model(train_loader, epochs, learning_rate)

# Evaluate classical network
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
accuracy = classical_network.evaluate_model(test_loader)

print(f"Accuracy of classical network: {accuracy * 100}%")

# Plot loss convergence
import matplotlib.pyplot as plt

plt.plot(classical_network.loss_history)
plt.title("Classical NN Training Convergence")
plt.xlabel("Training Iterations")

plt.ylabel("Binary Cross Entropy Loss")
plt.show()
