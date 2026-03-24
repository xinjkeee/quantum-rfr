import numpy as np
from time import time
from src.qrf import QRF

_qrf = None
_rf = None
_Y_MIN = None
_Y_RANGE = None
_t = None

def worker_init(rf_model, y_min, y_max, n_precision_qubits):
    """сall once per worker process at pool start"""
    global _qrf, _rf, _Y_MIN, _Y_RANGE, _t
    _rf = rf_model
    _Y_MIN = y_min
    _Y_RANGE = y_max - y_min
    _qrf = QRF(rf_model, y_min, y_max, n_precision_qubits)
    _t = 2 ** _qrf.n_amp
    

def process_single_sample(args):
    """fix pickling of QRF instance per call"""
    idx, x_sample, y_true, delta = args
    np.random.seed(int(time() * 1000) % 2**32 + idx)

    start_t = time()
    y_quantum, n_reps = _qrf.predict(x_sample, delta=delta)
    y_classic = _rf.predict(x_sample.reshape(1, -1))[0]
    duration = time() - start_t

    beta_exact = (y_classic - _Y_MIN) / _Y_RANGE
    beta_quantum = (y_quantum - _Y_MIN) / _Y_RANGE
    bound = 2 * np.pi * np.sqrt(beta_exact * (1 - beta_exact)) / _t + np.pi**2 / _t**2

    return {
        "idx": idx,
        "y_true": y_true,
        "y_classic": y_classic,
        "y_quantum": y_quantum,
        "beta_exact": beta_exact,
        "beta_quantum": beta_quantum,
        "bound": bound,
        "duration": duration
    }