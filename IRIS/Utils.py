import torch
from torch.nn import Module
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from tqdm import tqdm

def load_iris_dataset(binary_data: bool = True) -> list:
    """
    Load the filtered Iris dataset
    Returns the dataset as a list
    """
    iris = datasets.load_iris()
    
    class_labels = [0 if label == "setosa" else 1 if label == "versicolor" else 2 for label in iris.target_names]
    x = iris.data.tolist()
    y = iris.target.tolist()

    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)
    data = [[*x[i], class_labels[y[i]]] for i in range(len(x))]
    
    if binary_data:
        data = [row for row in data if row[-1] != 2]
        
    #print(data)
    return data

def transform_iris_data(data: list) -> tuple:
    """
    Transform the Iris dataset into input and target tensors
    Returns the input and target tensors
    """
    
    inputs = torch.tensor([row[:-1] for row in data], dtype=torch.float32)
    targets = torch.tensor([row[-1] for row in data], dtype=torch.float32).reshape(-1, 1)

    return inputs, targets

def get_data_loaders_from_labels(batch_size: int, train_size: float = 0.8, binary_data: bool = True) -> tuple:
    """
    Get data loaders for the Iris dataset with specified labels and number of samples
    """

    data = load_iris_dataset(binary_data)
    inputs, targets = transform_iris_data(data)

    dataset = torch.utils.data.TensorDataset(inputs, targets)
    train_size = int(train_size * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

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
            loss = criterion(outputs, labels)
            loss.backward()
            optimiser.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}, loss: {running_loss / len(train_loader)}")
        model.loss_history.append(running_loss / len(train_loader))
        
    model.trained = True
    
def evaluate_model(model: Module, test_loader: DataLoader) -> tuple[float, float]:
    """
    Evaluate the model using the specified data loader
    Returns accuracy, loss
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

            """print("----")
            print("Outputs: ")
            print(outputs)
            print("----")
            print("Labels: ")
            print(labels)"""

            pred = outputs.round()
            """print("----")
            print("Predictions: ")
            print(pred)"""
            correct += pred.eq(labels.view_as(pred)).sum().item()
            total += labels.size(0)

            loss = loss_func(outputs, labels)
            total_loss.append(loss.item())

        loss = sum(total_loss) / len(total_loss)
        accuracy = correct / total
        return accuracy, loss
    
def visualise_loss_history(model: Module):
    """
    Visualise the loss history of the model
    """
    
    plt.plot(model.loss_history)
    plt.title("Training Convergence")
    plt.xlabel("Training Iterations")
    plt.grid()
    plt.ylabel("BCE Loss")
    plt.show()

def visualise_loss_history_multiple(loss_histories: list[list], labels: list[str]):
    """
    Visualise the loss history of multiple models
    """
    
    for loss_history in loss_histories:
        plt.plot(loss_history)
    plt.title("Training Convergence")
    plt.xlabel("Training Iterations")
    # Include major gridlines
    plt.grid(visible=True, which='major', color='#666666', linestyle='-')
    # Customize minor gridlines
    plt.minorticks_on()
    plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.ylabel("BCE Loss")
    plt.legend(labels)
    plt.show()

def parameter_count(model: Module) -> int:
    """
    Get the number of parameters in the model
    """
    
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_model(model: Module, path: str):
    """
    Save the model to the specified path
    """
    torch.save(model.state_dict(), path)

def load_model(model: Module, path: str):
    """
    Load the model from the specified path
    """
    model.load_state_dict(torch.load(path))
    model.eval()
    model.trained = True

def save_loss_history(model: Module, path: str):
    """
    Save the loss history of the model to the specified path
    """
    np.save(path, model.loss_history)

def load_loss_history(model: Module, path: str) -> list:
    """
    Load the loss history of the model from the specified path
    """
    return np.load(path, allow_pickle=True).tolist()
