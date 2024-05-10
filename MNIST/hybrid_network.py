from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_algorithms.utils import algorithm_globals
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit.visualization import circuit_drawer

import torch
import torch.optim as optim
from torch.nn import (
    Module,
    Conv2d,
    Linear,
    Dropout2d,
    BCELoss,
)

def create_qnn():
    feature_map = ZZFeatureMap(2)
    ansatz = RealAmplitudes(2, reps=1)
    qc = QuantumCircuit(2)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)

    qnn = EstimatorQNN(
        circuit=qc,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        input_gradients=True
    )
    return qnn

class HybridNetwork(Module):
    def __init__(self, qnn):
        super().__init__()
        self.conv1 = Conv2d(1, 2, kernel_size=5)
        self.conv2 = Conv2d(2, 16, kernel_size=5)
        self.dropout = Dropout2d()
        self.fc1 = Linear(256, 64)
        self.fc2 = Linear(64, 2)
        self.qnn = TorchConnector(qnn)
        self.fc3 = Linear(1, 1)
        
        self.loss_function = BCELoss()

        self.loss_history = []
        self.trained = False

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = self.dropout(x)
        x = x.view(x.shape[0], -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.qnn(x)
        x = torch.sigmoid(self.fc3(x))
        return x
    
if __name__ == "__main__":
    import Utils
    epochs = 25
    learning_rate = 0.001
    batch_size = 10
    n_samples = 250
    train_loader, test_loader = Utils.get_data_loaders_from_labels([0, 1], n_samples, batch_size)
    
    qnn = create_qnn()
    print("Observables:", qnn.observables)
    print("Output shape:", qnn.output_shape)
    print("Number of input features:", qnn.num_inputs)
    print("Trainable parameters:", qnn.num_weights)
    print(circuit_drawer(qnn.circuit))
    
    model = HybridNetwork(qnn)
    
    optimiser = optim.Adam(model.parameters(), lr=learning_rate)
    
    loss_list = []  # Store loss history
    Utils.train_model(model, train_loader, epochs, learning_rate)
    
    accuracy = Utils.evaluate_model(model, test_loader)
    print(f"Accuracy of hybrid network: {accuracy * 100}%")
    