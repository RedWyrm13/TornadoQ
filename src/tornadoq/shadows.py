from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence, Tuple, Dict, Optional, Literal
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from .circuits import build_circuit


# ----------------------------
# Public configuration object
# ----------------------------
@dataclass
class ShadowConfig:
    """Configuration for collecting random-basis measurements."""
    T: int = 200                 # number of random-basis rounds per sample
    shots: int = 1024            # shots per round
    seed: Optional[int] = 1234   # seed for simulator + RNG (None => unseeded)
    backend: Optional[object] = None  # qiskit backend (defaults to AerSimulator)


# ----------------------------
# Random basis generation
# ----------------------------
def random_bases(n: int, rng: np.random.Generator) -> List[str]:
    """Sample a random local basis in {X,Y,Z}^n."""
    return rng.choice(["X", "Y", "Z"], size=n).tolist()


def add_measurement_layer(base: QuantumCircuit, bases: Sequence[str]) -> QuantumCircuit:
    """
    Compose `base` with per-qubit rotations to measure in the requested bases,
    then measure in the computational basis.
    """
    n = base.num_qubits
    qc = QuantumCircuit(n, n)
    qc.compose(base, inplace=True)
    for q, b in enumerate(bases):
        if b == "X":
            qc.h(q)            # X-basis via H then Z-measure
        elif b == "Y":
            qc.sdg(q); qc.h(q) # Y-basis via HS^† then Z-measure
        # if Z: do nothing
        qc.measure(q, q)
    return qc


# ----------------------------
# Collect shadows (measurements)
# ----------------------------
def collect_shadows(
    stateprep: QuantumCircuit,
    cfg: ShadowConfig = ShadowConfig(),
) -> Tuple[List[List[str]], List[np.ndarray]]:
    """
    Run T random local-basis rounds on `stateprep`.

    Returns
    -------
    bases_list : list length T
        Each element is a length-n list of 'X'/'Y'/'Z' strings.
    outcomes : list length T
        Each element is a (shots, n) array of ±1 measurement outcomes.
        Bit order is q0..q(n-1) across columns.
    """
    n = stateprep.num_qubits
    rng = np.random.default_rng(cfg.seed)
    bases_list = [random_bases(n, rng) for _ in range(cfg.T)]

    # Prepare backend
    backend = cfg.backend or AerSimulator(seed_simulator=cfg.seed)

    # Build circuits for all rounds
    circuits = [add_measurement_layer(stateprep, b) for b in bases_list]

    # Execute batched
    job = backend.run(circuits, shots=cfg.shots)
    result = job.result()

    outcomes: List[np.ndarray] = []
    for t in range(cfg.T):
        counts = result.get_counts(t)  # dict: bitstring -> count
        # Convert to matrix of ±1 with qubit columns in order q0..q(n-1)
        mats = []
        for bitstr, c in counts.items():
            # Qiskit returns classical bits c_(n-1)..c_0 in the string left..right.
            # We measured q -> c with same indices, so reverse to map to q0..q(n-1).
            bits = np.fromiter(bitstr[::-1], dtype=np.uint8)  # '0'/'1' bytes
            bits = bits - ord('0')                              # 0/1 ints
            pm = 1 - 2*bits                                    # 0 -> +1, 1 -> -1
            mats.append(np.tile(pm, (c, 1)))
        outcomes.append(np.vstack(mats) if mats else np.zeros((0, n), dtype=int))

    return bases_list, outcomes


# ----------------------------
# Pauli-string utilities
# ----------------------------
Pauli = Tuple[str, ...]  # e.g., ('Z','I','X',...)

def paulis_singles_xyz(n: int) -> List[Pauli]:
    """All single-qubit Paulis {X,Y,Z} on each of n qubits (size 3n)."""
    out: List[Pauli] = []
    for i in range(n):
        for a in ("X", "Y", "Z"):
            s = ["I"] * n
            s[i] = a
            out.append(tuple(s))
    return out


def paulis_ring_pairs(n: int, axes: Tuple[str, str] = ("Z", "Z")) -> List[Pauli]:
    """Ring pairs (i, i+1 mod n) with specified axes (e.g., ('Z','Z') => ZZ ring)."""
    out: List[Pauli] = []
    for i in range(n):
        j = (i + 1) % n
        s = ["I"] * n
        s[i], s[j] = axes
        out.append(tuple(s))
    return out


def paulis_all_pairs(n: int, axes: Tuple[str, str] = ("Z", "Z")) -> List[Pauli]:
    """
    All unordered pairs i<j with a fixed axis-pair (e.g., ('X','Y') => XY on every pair).
    Size: C(n,2)
    """
    a, b = axes
    out: List[Pauli] = []
    for i in range(n):
        for j in range(i + 1, n):
            s = ["I"] * n
            s[i], s[j] = a, b
            out.append(tuple(s))
    return out


def paulis_all_weight2(n: int, axes: Sequence[str] = ("X", "Y", "Z")) -> List[Pauli]:
    """All unordered pairs i<j with all axis combinations from `axes`."""
    out: List[Pauli] = []
    for i in range(n):
        for j in range(i + 1, n):
            for a in axes:
                for b in axes:
                    s = ["I"] * n
                    s[i], s[j] = a, b
                    out.append(tuple(s))
    return out  # size: C(n,2) * len(axes)^2  (e.g., 28 * 9 = 252 for n=8)


def label_of(P: Pauli) -> str:
    """Compact string label like 'ZIZX...'. Useful for column names."""
    return "".join(P)


# ----------------------------
# Estimation (classical shadows)
# ----------------------------
def estimate_pauli_expectations(
    bases_list: Sequence[Sequence[str]],
    outcomes: Sequence[np.ndarray],
    pauli_list: Sequence[Pauli],
) -> Dict[Pauli, float]:
    """
    Estimate <P> for each Pauli string P using Pauli-3 classical shadows.

    For a Pauli string P of weight w, a round is "compatible" iff every
    non-identity qubit of P was measured in the same axis. On compatible
    rounds, we multiply the ±1 outcomes on the support and scale by 3^w.
    Incompatible rounds contribute 0. Average across rounds.

    This yields an *unbiased* estimator for Pauli expectations under the
    Pauli-3 measurement scheme.

    Returns
    -------
    dict mapping P -> expectation estimate
    """
    assert len(bases_list) == len(outcomes), "Mismatched bases and outcomes."
    T = len(bases_list)

    # Precompute supports / weights
    supports: List[List[int]] = []
    weights: List[int] = []
    for P in pauli_list:
        supp = [i for i, a in enumerate(P) if a != "I"]
        supports.append(supp)
        weights.append(len(supp))

    feats: Dict[Pauli, float] = {}
    for idx, P in enumerate(pauli_list):
        supp = supports[idx]
        w = weights[idx]
        if w == 0:
            feats[P] = 1.0
            continue
        scale = 3 ** w

        acc = 0.0
        # Average over T rounds (incompatible ⇒ implicit 0 contribution)
        for t in range(T):
            bases = bases_list[t]
            if not all(bases[q] == P[q] for q in supp):
                continue
            m = outcomes[t][:, supp]  # (shots, w) of ±1
            if m.size == 0:
                continue
            prod = np.prod(m, axis=1)        # (shots,)
            acc += scale * float(np.mean(prod))
        feats[P] = acc / max(T, 1)

    return feats


# ----------------------------
# High-level: circuits -> matrix
# ----------------------------
def build_feature_matrix_from_circuits(
    circuits: Sequence[QuantumCircuit],
    pauli_list: Sequence[Pauli],
    cfg: ShadowConfig = ShadowConfig(),
) -> np.ndarray:
    """
    For each circuit, collect random-basis measurements and estimate features.
    Returns an array of shape (N_samples, N_features) in the order of `pauli_list`.
    """
    N = len(circuits)
    F = len(pauli_list)
    X = np.zeros((N, F), dtype=float)
    for k, circ in enumerate(circuits):
        bases, outs = collect_shadows(circ, cfg)
        feats = estimate_pauli_expectations(bases, outs, pauli_list)
        X[k, :] = [feats[P] for P in pauli_list]
    return X

# This can be removed since we handle normalization in the preprocess from preprocessing.py
def normalize_new_features(unnormalized_features):
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(feature_range=(0,1))

    normalized_features = scaler.fit_transform(unnormalized_features)
    return normalized_features
    

# I Need to change filesave_name to save no file when set to None. Currently, it saves to a default location
def generate_shadows(
    df,
    ring_paulis: List[str] = ['XY'],
    entanglement: str = 'ring',
    num_layers: int = 1,
    encoding_axis: Tuple[str, str] = ("rx", "ry"),
    train_test_val: Optional[str] = None,
    filename_save: Optional[str] = None,
    save: bool = False,
    pair_mode: Literal["ring", "all"] = "ring",
    T: int = 200,
    shots: int = 1000,
    seed: int = 123,
    # --- OPTIONAL: if True, use *all* 2-body combinations (XX..ZZ) instead of parsing ring_paulis ---
    all_weight2: bool = False,
):
    """
    df: pandas dataframe containing relevant data

    ring_paulis:
        List like ["XY", "ZZ"]. Each entry specifies which 2-body Pauli axis-pair to include.

    pair_mode:
        "ring" => only neighbor pairs (i, i+1 mod n)
        "all"  => all unordered pairs (i<j)

    all_weight2:
        If True, ignores ring_paulis and adds *all* 2-body Pauli strings (XX..ZZ) on the chosen pair_mode.
        - For pair_mode="ring": n pairs * 9 features
        - For pair_mode="all": C(n,2) pairs * 9 features

    T/shots/seed:
        Shadow sampling controls.
    """

    LABEL_COLS = ["ef_binary", "ef_class"]
    df_features = df.drop(columns=[c for c in LABEL_COLS if c in df.columns])
    data = df_features.to_numpy()

    # Gets size of feature space. We will use 1 qubit per feature
    n = len(data[0])

    # Builds a circuit for each datapoint
    circuits = [
        build_circuit(
            x,
            encoding_axes=encoding_axis,
            entanglement=entanglement,
            gate="cx",
            num_layers=num_layers,
        )
        for x in data
    ]

    # Always include single-qubit Paulis
    paulis: List[Pauli] = paulis_singles_xyz(n)

    # Add 2-body Paulis
    if all_weight2:
        if pair_mode == "all":
            paulis += paulis_all_weight2(n, axes=("X", "Y", "Z"))
        else:
            # ring-only but all axis combos
            # implement ring-weight2 by filtering all_weight2 to ring edges:
            ring_edges = {(i, (i + 1) % n) for i in range(n)}
            for i in range(n):
                j = (i + 1) % n
                for a in ("X", "Y", "Z"):
                    for b in ("X", "Y", "Z"):
                        s = ["I"] * n
                        s[i], s[j] = a, b
                        paulis.append(tuple(s))
    else:
        # Parse ring_paulis entries like "XY" and add either ring pairs or all pairs for that axis-pair
        for pair in ring_paulis:
            pair = pair.upper().strip()
            if len(pair) != 2 or any(p not in {"I", "X", "Y", "Z"} for p in pair):
                raise ValueError(f"Invalid Pauli pair '{pair}'. Expected like 'XY', 'ZZ', etc.")
            axes = (pair[0], pair[1])
            if pair_mode == "all":
                paulis += paulis_all_pairs(n, axes)
            else:
                paulis += paulis_ring_pairs(n, axes)

    # Shadow measurement parameters
    cfg = ShadowConfig(T=T, shots=shots, seed=seed)

    # Generate features
    new_features = build_feature_matrix_from_circuits(circuits, paulis, cfg)

    # Normalize (optional; you said preprocess can handle this)
    normalized_features = normalize_new_features(new_features)

    # Build output df (keep column labels if you want)
    out_df = pd.DataFrame(normalized_features, columns=[label_of(P) for P in paulis])

    # Save logic: ONLY save if explicitly requested
    if filename_save is not None:
        out_df.to_csv(filename_save, index=False)
    elif save:
        rings_str = "ALLW2" if all_weight2 else "_".join(ring_paulis)
        filename_save = (
            f"../{normalized_features.shape[1]}_features_"
            f"{rings_str}_{pair_mode}_{train_test_val}_QuantumLayers{num_layers}.csv"
        )
        out_df.to_csv(filename_save, index=False)

    return out_df


# Meant to combine the original features with the shadow features
def extend_features(df_1, df_2):
    df = pd.concat([df_1, df_2], axis=1)
    return df
