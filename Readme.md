# Quantum Random Forest for Regression

This repository contains a Quantum Random Forest (QRF) implementation for regression. The classical Random Forest is trained with scikit-learn; PennyLane is used to simulate the Quantum Amplitude Estimation (QAE) aggregation step.

## Repository contents

- `main.py` — experiment entry point;
- `src/qrf.py` — QRF circuit and QAE prediction logic;
- `src/utils.py` — multiprocessing helpers;
- `data/ENB2012_data.csv` — Energy Efficiency dataset;
- `requirements.txt` — Python dependencies.

## Dataset

The ENB2012 dataset contains 768 observations, 8 input features, and two targets:

- `Y1` — heating load;
- `Y2` — cooling load.

The standard experiment uses an 80/20 split: 614 training and 154 test observations, with `random_state=42`.

## Classical Random Forest configuration

- `n_estimators=128`;
- `max_depth=6`;
- `criterion='squared_error'`;
- `random_state=42`;
- bootstrap sampling enabled by the scikit-learn default.

The quantum model uses the same trained forest. QAE is applied during inference to approximate the ensemble prediction.

## Quantum simulation

The implementation uses:

- PennyLane;
- `pennylane-lightning`;
- `lightning.qubit` CPU state-vector simulation;

`N_PRECISION_QUBITS` is the QAE precision register size. For example, `m=5` gives `2^5=32` QAE grid points. The original register-based implementation also contains a tree-path register. The compact experimental variant used in the latest m=9 runs omits this register and uses 17 qubits in total for 128 trees and depth 6.

## Reproducibility

The dataset split and classical forest use seed `42`. The latest server runner uses deterministic per-object NumPy seeds derived from the experiment seed and test-object index. Generated results, logs, virtual environments, server scripts, and figures are intentionally excluded from Git by `.gitignore`.

## Hardware used for the latest experiment

The VAST instance used for the latest experiment was:

- CPU: AMD EPYC 7B12 64-Core Processor;
- allocation shown by VAST: `64/128 CPU`;
- RAM shown by VAST: `64.3 GB`;
- GPU: NVIDIA GeForce RTX 5080;
- VRAM shown by VAST: `15.9 GB`;

The RTX 5080 was not used: the selected `lightning.qubit` backend
performs CPU state-vector simulation.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

The default `main.py` is the original research entry point. For production-scale server runs, use the separately maintained server runner outside this minimal algorithm repository.

## Citation and limitations

This implementation is a hybrid CPU simulation intended to study QAE-based aggregation of a classical Random Forest. Its simulator wall time should not be interpreted as a hardware quantum speedup. Percentage metrics such as MAPE, wMAPE, and sMAPE evaluated on the original physical load, normalized `beta` evaluated with absolute metrics such as MAE and RMSE.
