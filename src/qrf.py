"""Compact QAE aggregation of a classically trained Random Forest"""
import math

import numpy as np
import pennylane as qml


def get_alpha(prediction, y_min, y_max):
    norm = np.clip((prediction - y_min) / (y_max - y_min + 1e-15), 0, 1)
    return np.arcsin(np.sqrt(norm))


def reflection_R(wires):
    """Reflection I - 2|0><0| on the tree and output registers."""
    for wire in wires:
        qml.PauliX(wires=wire)
    qml.Hadamard(wires=wires[-1])
    qml.MultiControlledX(wires=wires)
    qml.Hadamard(wires=wires[-1])
    for wire in wires:
        qml.PauliX(wires=wire)


def repetitions_for_delta(delta):
    """Odd median repetitions certified by the generic QAE success bound."""
    if not 0 < delta < 0.5:
        raise ValueError("delta must be between 0 and 0.5")
    failure = 1 - 8 / math.pi ** 2
    for count in range(1, 1001, 2):
        bound = sum(math.comb(count, k) * failure ** k * (1 - failure) ** (count - k)
                    for k in range(count // 2 + 1, count + 1))
        if bound <= delta:
            return count
    raise ValueError("delta is too small")


class QRF:
    def __init__(self, rf_model, y_min, y_max, n_precision_qubits=9):
        self.rf = rf_model
        self.y_min = float(y_min)
        self.y_max = float(y_max)
        if not np.isfinite([self.y_min, self.y_max]).all() or y_max <= y_min:
            raise ValueError("Normalization requires finite y_max > y_min")
        if not isinstance(n_precision_qubits, int) or n_precision_qubits < 1:
            raise ValueError("n_precision_qubits must be a positive integer")

        self.n_trees = len(rf_model.estimators_)
        if self.n_trees < 2 or self.n_trees & (self.n_trees - 1):
            raise ValueError("Tree count must be a power of two, at least 2")
        self.tree_depth = max(tree.get_depth() for tree in rf_model.estimators_)
        self.n_lambda = int(math.log2(self.n_trees))
        self.lambda_wires = list(range(self.n_lambda))
        self.phi_wire = self.n_lambda
        self.n_amp = n_precision_qubits
        self.amp_wires = list(range(self.phi_wire + 1, self.phi_wire + 1 + self.n_amp))
        self.total_wires = self.amp_wires[-1] + 1
        self.tree_data = []

    def _tree_control_values(self, tree_idx):
        return [int(bit) for bit in format(tree_idx, f"0{self.n_lambda}b")]

    def prepare_tree_data(self, X_input):
        sample = np.asarray(X_input).reshape(1, -1)
        self.tree_data = [
            {"alpha": get_alpha(tree.predict(sample)[0], self.y_min, self.y_max)}
            for tree in self.rf.estimators_
        ]

    def U_operator(self):
        for wire in self.lambda_wires:
            qml.Hadamard(wires=wire)
        for tree_idx, data in enumerate(self.tree_data):
            qml.ctrl(qml.RY, control=self.lambda_wires,
                     control_values=self._tree_control_values(tree_idx))(
                         2 * data["alpha"], wires=self.phi_wire)

    def grover_step(self):
        qml.PauliX(wires=self.phi_wire)
        qml.PauliZ(wires=self.phi_wire)
        qml.PauliX(wires=self.phi_wire)
        qml.adjoint(self.U_operator)()
        reflection_R(self.lambda_wires + [self.phi_wire])
        self.U_operator()

    def get_circuit_probs(self, X_input):
        self.prepare_tree_data(X_input)
        dev = qml.device("lightning.qubit", wires=self.total_wires, shots=None)

        @qml.qnode(dev)
        def circuit():
            self.U_operator()
            for wire in self.amp_wires:
                qml.Hadamard(wires=wire)
            for i, control_wire in enumerate(self.amp_wires):
                for _ in range(2 ** (self.n_amp - 1 - i)):
                    qml.ctrl(self.grover_step, control=control_wire)()
            qml.adjoint(qml.QFT)(wires=self.amp_wires)
            return qml.probs(wires=self.amp_wires)

        return circuit()

    def predict(self, X_input, delta=0.05, rng=None):
        """Sample the simulated distribution and return the median load estimate"""
        count = repetitions_for_delta(delta)  # 7 for delta=0.05
        probs = np.asarray(self.get_circuit_probs(X_input), dtype=float)
        if (not np.isfinite(probs).all() or probs.min() < -1e-12
                or not np.isclose(probs.sum(), 1, atol=1e-8, rtol=0)):
            raise ValueError("Invalid circuit probability distribution")
        probs = np.maximum(probs, 0)
        probs /= probs.sum()
        rng = np.random.default_rng() if rng is None else rng
        measured = rng.choice(len(probs), size=count, p=probs)
        beta = float(np.median(np.sin(np.pi * measured / (2 ** self.n_amp)) ** 2))
        return beta * (self.y_max - self.y_min) + self.y_min, count
