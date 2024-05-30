from classical_network import ClassicalNetwork
from hybrid_network import create_qnn, HybridNetwork
import torch
import Utils

torch.manual_seed(39)

train_loader, test_loader = Utils.get_data_loaders_from_labels(batch_size=10)

classical_network = ClassicalNetwork()
Utils.train_model(classical_network, train_loader, epochs=100, learning_rate=0.001)
accuracy, loss = Utils.evaluate_model(classical_network, test_loader)

Utils.visualise_loss_history(classical_network)

print(f"Accuracy of classical network: {accuracy * 100}%")
print(f"Loss of classical network: {loss}")
print(f"Model parameters: {Utils.parameter_count(classical_network)}")

Utils.save_loss_history(classical_network, "classical_loss_history")
Utils.save_model(classical_network, "classical_model.pth")