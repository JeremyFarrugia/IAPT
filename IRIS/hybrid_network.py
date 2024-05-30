from torch import Tensor, relu, sigmoid
from torch.nn import Module, Linear, BCELoss

from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_algorithms.utils import algorithm_globals
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit.visualization import circuit_drawer, plot_gate_map
from qiskit_machine_learning.algorithms import NeuralNetworkClassifier

from matplotlib import pyplot as plt

def create_qnn(num_qubits: int = 4, feature_map_depth: int = 2, ansatz_depth: int = 2) -> EstimatorQNN:
    """
    Create a quantum neural network with the specified number of qubits, feature map depth, and ansatz depth
    ZZFeatureMap and RealAmplitudes are used for the feature map and ansatz, respectively
    Both are set to full entanglement
    """
    

    feature_map = ZZFeatureMap(num_qubits, reps=feature_map_depth, entanglement="full")
    ansatz = RealAmplitudes(num_qubits, reps=ansatz_depth, entanglement="full")
    qc = QuantumCircuit(num_qubits)
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
    """
    Network composed of a quantum circuit and a classical neural network
    """
    
    def __init__(self, quantum_nn: EstimatorQNN):
        super(HybridNetwork, self).__init__()
        self.fc1 = Linear(4, 12)
        self.fc2 = Linear(12, 2)
        self.qnn = TorchConnector(quantum_nn)
        self.fc4 = Linear(1, 1)
        
        self.loss_function = BCELoss()
        
        self.loss_history = []
        self.trained = False
        
    def forward(self, x: Tensor) -> Tensor:
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = self.qnn(x)
        x = sigmoid(self.fc4(x))
        return x
    
if __name__ == "__main__":
    import Utils
    import torch
    
    torch.manual_seed(39)
    #algorithm_globals.random_seed = 39

    train_loader, test_loader = Utils.get_data_loaders_from_labels(batch_size=10)

    circ = create_qnn(num_qubits=2, feature_map_depth=2, ansatz_depth=3)

    hybrid_network = HybridNetwork(circ)

    Utils.train_model(hybrid_network, train_loader, epochs=100, learning_rate=0.001)
    accuracy, loss = Utils.evaluate_model(hybrid_network, test_loader)

    Utils.visualise_loss_history(hybrid_network)

    print(f"Accuracy of hybrid network: {accuracy * 100}%")
    print(f"Loss of hybrid network: {loss}")
    print(f"Model parameters: {Utils.parameter_count(hybrid_network)}")

    Utils.save_loss_history(hybrid_network, "hybrid_loss_history")
    Utils.save_model(hybrid_network, "hybrid_model.pth")