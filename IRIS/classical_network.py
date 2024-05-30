import torch
from torch.nn import Module, Conv2d, Linear, Dropout2d

class ClassicalNetwork(Module):
    """
    Classical neural network for binary classification of setosa and versicolor from the Iris dataset
    Used as a comparison to the quantum neural network
    """
    
    def __init__(self):
        super(ClassicalNetwork, self).__init__()
        self.fc1 = Linear(4, 12)
        self.fc2 = Linear(12, 2)
        self.fc3 = Linear(2, 1)

        self.loss_function = torch.nn.BCELoss()

        self.loss_history = []
        self.trained = False
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict the output of the network for the given input
        """
        self.eval()
        with torch.no_grad():
            return self(x).squeeze(1).float()
    
    def predict_class(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict the class of the network for the given input
        """
        return self.predict(x).round()
