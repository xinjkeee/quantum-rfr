import numpy as np
import pennylane as qml
from numpy import binary_repr

def get_alpha(prediction, y_min, y_max):

    norm = np.clip((prediction - y_min) / (y_max - y_min + 1e-15), 0, 1)
    alpha = np.arcsin(np.sqrt(norm))

    return alpha


def get_tree_path_bits(X_input, tree, max_depth):
    node = tree.decision_path(X_input.reshape(1, -1))
    node_index = node.indices[node.indptr[0]: node.indptr[1]]
    
    path_bits =[]

    for i in range(len(node_index) - 1):
        curr_node = node_index[i]
        next_node = node_index[i+1]

        if tree.tree_.children_right[curr_node] == next_node:
            path_bits.append(1)
        else:
            path_bits.append(0)

        
    if len(path_bits) < max_depth:
        path_bits += [0] * (max_depth - len(path_bits))

    return path_bits[:max_depth]

def multiplication(wires):
    for i in range(len(wires) - 1):
        qml.SWAP(wires=[wires[i], wires[i+1]])


# I - 2|fi><fi>
def reflection_R(wires):
    for w in wires:
        qml.PauliX(wires=w)
    
    qml.Hadamard(wires=wires[-1])
    qml.MultiControlledX(wires=wires)
    qml.Hadamard(wires=wires[-1])

    for w in wires:
        qml.PauliX(wires=w)

class QRF:
    def __init__(self, rf_model, y_min, y_max, n_precision_qubits=4):
        self.rf = rf_model
        self.y_min = y_min
        self.y_max = y_max

        self.n_trees = len(rf_model.estimators_)
        self.tree_depth = max([e.get_depth() for e in rf_model.estimators_])

        self.n_lambda = int(np.ceil(np.log2(self.n_trees)))
        self.lambda_wires = list(range(self.n_lambda))

        self.n_psi = self.tree_depth + 1
        self.psi_wires = list(range(self.n_lambda, self.n_psi + self.n_lambda))

        self.phi_wire = self.psi_wires[-1] + 1

        self.n_amp = n_precision_qubits
        self.amp_wires = list(range(self.phi_wire + 1, self.phi_wire + 1 + self.n_amp))


        self.total_wires = self.amp_wires[-1] + 1

        self.tree_data =[]

    def _tree_control_values(self, tree_idx):
        return[int(x) for x in list(binary_repr(tree_idx, self.n_lambda))]

    def prepare_tree_data(self, X_input):
        self.tree_data =[] 
        for i, tree in enumerate(self.rf.estimators_):
            bits = get_tree_path_bits(X_input, tree, tree.max_depth)         

            pred = tree.predict(X_input.reshape(1, -1))[0]
            alpha = get_alpha(pred, self.y_min, self.y_max)

            self.tree_data.append({
                'id': i,
                'path_bits': bits,
                'alpha': alpha
            })


    def U_operator(self):
        for wire in self.lambda_wires:
            qml.Hadamard(wires=wire)
        
        qml.PauliX(wires=self.psi_wires[-1])


        for depth in range(self.tree_depth):
            for tree_idx in range(self.n_trees):

                if depth < len(self.tree_data[tree_idx]['path_bits']):
                    control_values = self._tree_control_values(tree_idx)
                    direction_bit = self.tree_data[tree_idx]['path_bits'][depth]

                    qml.ctrl(multiplication,control=self.lambda_wires, control_values=control_values)(self.psi_wires)


                    if direction_bit == 1:
                        qml.MultiControlledX(wires=self.lambda_wires + [self.psi_wires[-1]], control_values=control_values)
                    
        
        for tree_idx in range(self.n_trees):
            control_values = self._tree_control_values(tree_idx)
            alpha = self.tree_data[tree_idx]['alpha']

            qml.ctrl(qml.RY, control=self.lambda_wires, control_values=control_values)(2*alpha, wires=self.phi_wire)

    def grover_step(self):
        qml.PauliX(wires=self.phi_wire) # XZX = -Z; I - 2|fi><fi| -----> 2|fi><fi> - I
        qml.PauliZ(wires=self.phi_wire)
        qml.PauliX(wires=self.phi_wire)
        qml.adjoint(self.U_operator)()
        reflection_R(self.lambda_wires + self.psi_wires + [self.phi_wire])
        self.U_operator()
    
    def get_circuit_probs(self, X_input):
        self.prepare_tree_data(X_input)

        dev = qml.device("lightning.qubit", wires=self.total_wires)

        @qml.qnode(device=dev)
        def circuit():
            
            self.U_operator()

            for w in self.amp_wires: 
                qml.Hadamard(wires=w)

            for i, control_wire in enumerate(self.amp_wires):
                power = 2**(self.n_amp - 1 - i)
                for _ in range(power):
                    qml.ctrl(self.grover_step, control=control_wire)()

            qml.adjoint(qml.QFT)(wires=self.amp_wires)
            
            return qml.probs(wires=self.amp_wires)

        return circuit()

    def predict(self, X_input, delta=0.05):
        n_reps = max(1, int(np.ceil(np.log(1.0 / delta))))

        probs = self.get_circuit_probs(X_input)
        m = self.n_amp
        k_values = np.arange(2 ** m)

        measured_ks = np.random.choice(k_values, size=n_reps, p=probs) # sample k values according to the probabilities from the circuit
        beta_estimates = np.sin(np.pi * measured_ks / (2 ** m)) ** 2 # beta = sin^2(pi * k / 2^m)
        y_estimates = beta_estimates * (self.y_max - self.y_min) + self.y_min # convert beta back to the original

        return np.median(y_estimates), n_reps