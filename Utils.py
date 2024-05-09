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

def train_model(model: Module, train_loader: DataLoader, epochs: int, learning_rate: float):
    """
    Train the network using the specified data loader, number of epochs, and learning rate
    Optimiser: Adam
    Loss function: Negative Log Likelihood Loss
    """

    if model.trained:
        print("Model has already been trained. To retrain, create a new model.")
        return

    optimiser = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = model.loss_function
    model.train() # Set model to training mode

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(tqdm(train_loader,desc=f"Epoch {epoch + 1}")):
            inputs, labels = data
            optimiser.zero_grad()
            outputs = model(inputs)
            labels = labels.unsqueeze(1).float()
            loss = criterion(outputs, labels)
            loss.backward()
            optimiser.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}, loss: {running_loss / len(train_loader)}")
        model.loss_history.append(running_loss / len(train_loader))
        
    model.trained = True
            
def predict(model: Module, x: torch.Tensor) -> torch.Tensor:
    """
    Predict the output of the network for the given input
    """
    model.eval()
    with torch.no_grad():
        return model(x).squeeze(1).float()
    
def predict_class(model: Module, x: torch.Tensor) -> torch.Tensor:
    """
    Predict the class of the network for the given input
    """
    return model.predict(x).round()

def evaluate_model(model: Module, test_loader: DataLoader) -> float:
    """
    Evaluate the model using the specified data loader
    Returns the accuracy of the model
    """

    loss_func = model.loss_function

    model.eval()
    correct = 0
    total = 0

    total_loss = []

    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            outputs = model(inputs)
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
    