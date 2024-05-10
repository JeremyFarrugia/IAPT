from torch import Tensor, relu, sigmoid
from torch.nn import Module, Linear, BCELoss

from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_algorithms.utils import algorithm_globals
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit.visualization import circuit_drawer
from qiskit_machine_learning.algorithms import NeuralNetworkClassifier

def create_qnn():
    """
    Note that the output of the quantum circuit is a single value, which is then passed through a sigmoid function
    """
    feature_map = ZZFeatureMap(2, reps=5, entanglement="full")
    ansatz = RealAmplitudes(2, reps=1, entanglement="full")
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
    """
    Network composed of a quantum circuit and a classical neural network
    """
    
    def __init__(self, quantum_nn: EstimatorQNN):
        super(HybridNetwork, self).__init__()
        self.fc1 = Linear(4, 24)
        self.fc2 = Linear(24, 4)
        self.fc3 = Linear(4, 2)
        self.qnn = TorchConnector(quantum_nn)
        self.fc4 = Linear(1, 1)
        
        self.loss_function = BCELoss()
        
        self.loss_history = []
        self.trained = False
        
    def forward(self, x: Tensor) -> Tensor:
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = relu(self.fc3(x))
        x = self.qnn(x)
        x = sigmoid(self.fc4(x))
        return x
    
if __name__ == "__main__":
    import Utils
    import torch
    

    train_loader, test_loader = Utils.get_data_loaders_from_labels(batch_size=10)

    circ = create_qnn()
    print(circuit_drawer(circ.circuit))
    print(circuit_drawer(circ.circuit.decompose()))
    print(circ.num_weights)
    hybrid_network = HybridNetwork(circ)
    Utils.train_model(hybrid_network, train_loader, epochs=100, learning_rate=0.001)
    accuracy, loss = Utils.evaluate_model(hybrid_network, test_loader)

    Utils.visualise_loss_history(hybrid_network)

    print(f"Accuracy of classical network: {accuracy * 100}%")
    print(f"Loss of classical network: {loss}")