import torch
from torch.nn import Module, Conv2d, Linear, Dropout2d

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
