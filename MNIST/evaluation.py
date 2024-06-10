# Load the loss histories
import numpy as np
import matplotlib.pyplot as plt

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

classical = np.load("mnist_loss_history.npy")
quantum = np.load("mnist_qnn_loss_history.npy")
visualise_loss_history_multiple([classical, quantum], ["Classical", "Quantum"])