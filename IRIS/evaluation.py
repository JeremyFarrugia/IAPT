from classical_network import ClassicalNetwork
from hybrid_network import create_qnn, HybridNetwork
import Utils

# Load loss history
classical_loss_history = Utils.load_loss_history(ClassicalNetwork(), "classical_loss_history.npy")
hybrid_loss_history = Utils.load_loss_history(HybridNetwork(create_qnn()), "hybrid_loss_history.npy")

# Visualise loss history
Utils.visualise_loss_history_multiple([classical_loss_history, hybrid_loss_history], ["Classical", "Hybrid"])