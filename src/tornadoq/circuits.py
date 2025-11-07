from typing import Iterable, Literal, Sequence, Tuple
import numpy as np
from qiskit import QuantumCircuit

# ------------------------------------------------------------
# Public entry point: build a circuit that (1) encodes features
# and (2) applies a configurable entangling block.
# ------------------------------------------------------------
def build_circuit(
    data: Iterable[float],
    *,
    # ---- Encoding controls ----
    normalize: bool = True,
    angle_scale: float = np.pi,                  # multiply angles by this (e.g., π)
    encoding_axes: Sequence[str] = ("rx", "ry"), # which rotations to use per feature
                                                 # e.g., ("ry",) or ("ry","rz") etc.
    # ---- Entangler controls ----
    entanglement: Literal["full", "ring", "linear"] = "full",
    gate:        Literal["cx", "cz"] = "cx",
    num_layers:  int = 2,
    alternate_directions: bool = True,
    add_barriers: bool = True,
) -> QuantumCircuit:
    """
    Builds a feature-generating circuit for your shadows pipeline:
      data -> [angle encoding] -> [entangling layers]

    Parameters
    ----------
    data : Iterable[float]
        Your feature vector (one qubit per feature).
    normalize : bool
        Whether to apply tanh normalization (maps to [-1,1]) before angle scaling.
        Recommended for robustness.
    angle_scale : float
        Scalar to multiply features when used as rotation angles (default π).
    encoding_axes : Sequence[str]
        Rotations to apply per feature, in order. Options: "rx", "ry", "rz".
        Example: ("ry",) or ("ry","rz") or ("rx","ry","rz").
    entanglement : {"full","ring","linear"}
        Connectivity of the entangling block.
    gate : {"cx","cz"}
        Two-qubit gate family. Use "cz" if your backend favors symmetric CZ.
    num_layers : int
        Number of ansatz layers. We use data reuploading.
    alternate_directions : bool
        If using CX, flip control/target each layer to reduce directional bias.
    add_barriers : bool
        Add visual/compile barriers between layers (useful during debugging).

    Returns
    -------
    QuantumCircuit
        Circuit with encoding + entanglement applied.
    """
    data = np.asarray(list(data), dtype=float)
    n = int(data.size)
    assert n >= 2, "Need at least 2 qubits (features) to add entanglement."

    # ---------- 1) Normalize & scale angles ----------
    # Good default for continuous geophysical features (robust to outliers).
    if normalize:
        data = np.tanh(data)               # maps to [-1,1]
    thetas = angle_scale * data            # rescale to angles

    # ---------- 2) Encoding ----------
    qc = QuantumCircuit(n, name="encode+entangle")

    for layer in range(num_layers):
        _apply_angle_encoding(qc, thetas, encoding_axes)

        # ---------- 3) Entangling block ----------
        _add_entangling_layer(
            qc,
            entanglement=entanglement,
            gate=gate,
            alternate_directions=alternate_directions,
        )
        if add_barriers:
            qc.barrier()
    return qc


# ------------------------------------------------------------
# HELPER: angle-encoding with flexible axes per feature
# ------------------------------------------------------------
def _apply_angle_encoding(
    qc: QuantumCircuit,
    thetas: np.ndarray,
    axes: Sequence[str],
) -> None:
    """
    Apply angle encoding per feature/qubit.

    Each feature angle θ_i is applied using the sequence of rotations
    specified in `axes` (e.g., ("ry","rz") applies RY(θ_i) then RZ(θ_i)).
    """
    valid = {"rx", "ry", "rz"}
    axes = tuple(ax.lower() for ax in axes)
    if not axes:
        raise ValueError("encoding_axes must contain at least one of {'rx','ry','rz'}.")
    if any(ax not in valid for ax in axes):
        raise ValueError(f"encoding_axes must be in {valid}, got {axes}.")

    for q, theta in enumerate(thetas):
        for ax in axes:
            if ax == "rx":
                qc.rx(theta, q)
            elif ax == "ry":
                qc.ry(theta, q)
            elif ax == "rz":
                qc.rz(theta, q)


# ------------------------------------------------------------
# HELPER: entangling layer with full/ring/linear topologies
# ------------------------------------------------------------
def _add_entangling_layer(
    qc: QuantumCircuit,
    *,
    entanglement: Literal["full", "ring", "linear"] = "full",
    gate: Literal["cx", "cz"] = "cx",
    alternate_directions: bool = True,
) -> None:
    """

    Topologies
    ----------
    full   : all unordered pairs (i < j)
    ring   : (i, i+1 mod n)     # wrap-around; degree-2 each
    linear : (i, i+1)           # no wrap; good for chains or 1D hardware

    Gate choice
    -----------
    cx : directional; we optionally flip directions on odd layers to avoid bias.
    cz : symmetric; directionless (useful if backend natively supports CZ).
    """
    n = qc.num_qubits
    assert n >= 2, "Need at least 2 qubits to entangle."

    # Build pair list once (deduplicated and safe).
    pairs = _pairs_for_topology(n, entanglement)


    for (i, j) in pairs:
        if gate == "cx":
            # Flip control/target every other layer to share control load,
            # which can help mitigate readout and direction-specific noise.
            c, t = (i, j)
            qc.cx(c, t)
        elif gate == "cz":
            # CZ is symmetric; no need to flip.
            qc.cz(i, j)
        else:
            raise ValueError("gate must be 'cx' or 'cz'.")




def _pairs_for_topology(
    n: int,
    kind: Literal["full", "ring", "linear"],
) -> Sequence[Tuple[int, int]]:
    if kind == "full":
        # All-to-all once, no self-edges, i<j prevents duplicates.
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    if kind == "ring":
        # Degree-2 ring; last connects to first via modulo.
        return [(i, (i + 1) % n) for i in range(n)]
    if kind == "linear":
        # Open chain; no wrap-around.
        return [(i, i + 1) for i in range(n - 1)]
    raise ValueError("entanglement must be 'full', 'ring', or 'linear'")

def customized_entanglement(qc, indices):
    
    for x,y in indices:
        qc.cx(x,y)
        
    return qc
