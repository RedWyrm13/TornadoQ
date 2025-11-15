import pennylane as qml
import numpy as np
import torch
from pennylane.templates import RandomLayers


def InitializePQC(circuit):
    # Quantum device and parameters
    n_qubits = 8
    n_layers = 1
    dev = qml.device("default.qubit", wires=n_qubits)
    
    # Using Random Layers PQC
    if circuit == "RandomLayer":
    
        # RandomLayers expects shape (n_layers, n_rotations_per_layer)
        rand_params = np.random.uniform(0, 2 * np.pi, (n_layers, n_qubits * 3))
    
        @qml.qnode(dev, interface="torch", diff_method="backprop")
        quantum_feature_embedding = RL_feature_embedding
    
    # Using Strongly Entangling Layers
    if circuit == "StronglyEntangled":

        # StronglyEntanglingLayers expects shape (n_layers, n_wires, 3)
        rand_params = np.random.uniform(0, 2 * np.pi, (n_layers, n_qubits, 3))
    
        @qml.qnode(dev, interface="torch", diff_method="backprop")
        quantum_feature_embedding = SE_feature_embedding
        
        
    ##### Other PQCS here #####


    # Batched embedding helper
    def quantum_feature_embedding_batch(x_batch, phi, device="cuda"):
        """Apply QNode to batch of inputs -> (B, 256)"""
        outputs = []
        phi = phi.detach().cpu()
        for x in x_batch:
            result = quantum_feature_embedding(x.detach().cpu(), phi)
            outputs.append(result.real.to(device))
        return torch.stack(outputs)
    
    
    # Torch module for integration
    class QuantumFeatureEmbeddingBatch(nn.Module):
        def __init__(self, device="cuda"):
            super().__init__()
            self.device = device
            self.phi = nn.Parameter(torch.tensor(
                np.random.uniform(0, 2 * np.pi, (n_layers, n_qubits * 3)), 
                dtype=torch.float32
            ))
    
        def forward(self, x_batch):
            outputs = []
            phi = self.phi.detach().cpu()
            for x in x_batch:
                result = quantum_feature_embedding(x.detach().cpu(), phi)
                outputs.append(result.real.to(self.device))
            return torch.stack(outputs)

def RL_feature_embedding(f, phi):
            qml.AngleEmbedding(features=f, wires=range(n_qubits))
            qml.RandomLayers(phi, wires=range(n_qubits), seed=6)
            return qml.state()  # returns 2**n_qubits = 256-dim statevector

def SE_feature_embedding(f, phi):
            qml.AngleEmbedding(features=f, wires=range(n_qubits))
            qml.StronglyEntanglingLayers(weights=phi, wires=range(n_qubits))
            return qml.state()

