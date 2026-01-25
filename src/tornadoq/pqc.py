import pennylane as qml
import numpy as np
import torch
import torch.nn as nn
from pennylane.templates import RandomLayers, StronglyEntanglingLayers
from tornadoq.helper import resolve_device

# ---------------------------------------------------
# Feature embedding circuits
# ---------------------------------------------------
def RL_feature_embedding(f, phi, n_qubits):
    qml.AngleEmbedding(features=f, wires=range(n_qubits))
    qml.RandomLayers(phi, wires=range(n_qubits), seed=6)
    return qml.state()


def SE_feature_embedding(f, phi, n_qubits):
    qml.AngleEmbedding(features=f, wires=range(n_qubits))
    qml.StronglyEntanglingLayers(weights=phi, wires=range(n_qubits))
    return qml.state()


# ---------------------------------------------------
# Main initializer
# ---------------------------------------------------
def InitializePQC(circuit, device):
    n_qubits = 8
    n_layers = 1
    dev = qml.device("default.qubit", wires=n_qubits)

    # ---------------------------------------------
    # Select PQC type
    # ---------------------------------------------
    if circuit == "RandomLayer":
        phi_shape = (n_layers, n_qubits * 3)
        
        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def qnode(f, phi):
            return RL_feature_embedding(f, phi, n_qubits)

    elif circuit == "StronglyEntangling":
        phi_shape = (n_layers, n_qubits, 3)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def qnode(f, phi):
            return SE_feature_embedding(f, phi, n_qubits)

    else:
        raise ValueError("circuit must be 'RandomLayer' or 'StronglyEntangling'")

    # ------------------------------------------------
    # Torch module wrapper for batching
    # ------------------------------------------------
    class QuantumFeatureEmbedding(nn.Module, device):
        def __init__(self):
            super().__init__()
            self.device = resolve_device(device)
            
            # Trainable PQC parameters
            self.phi = nn.Parameter(
                torch.tensor(
                    np.random.uniform(0, 2 * np.pi, phi_shape),
                    dtype=torch.float32
                )
            )

        def forward(self, x_batch):
            """Compute PQC outputs for batch -> (B, 256)"""
            out = []
            phi_cpu = self.phi.detach().cpu()

            for x in x_batch:
                result = qnode(x.detach().cpu(), phi_cpu)
                out.append(result.real.to(self.device))

            return torch.stack(out)

    return QuantumFeatureEmbedding()

