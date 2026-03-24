# Quantum Random Forest for Regression Problem

This repository provides a Python implementation of the Quantum Random Forest (QRF) algorithm for regression tasks. It simulates the Quantum Amplitude Estimation (QAE) process to approximate the results of a classical Random Forest model.

## Structure
- `data/`: Contains the dataset (e.g., ENB2012_data.csv).
- `src/qrf.py`: Contains the `QRF` quantum class and fundamental operations.
- `src/utils.py`: Utilities for multiprocessing execution.
- `main.py`: The main entry point to run the regression simulations.

## Installation
```bash
pip install -r requirements.txt
```

## Dataset
The project uses the **Energy Efficiency Dataset (ENB2012)** from the UCI Machine Learning Repository. It contains 768 samples with 8 features to predict two real-valued responses:
- **Y1**: Heating Load
- **Y2**: Cooling Load

## Usage
To run the simulation for both targets (Y1 and Y2), one can execute:
```bash
python main.py
```

## Configuration
One can adjust the simulation parameters directly in `main.py` to match hardware capabilities:

- `N_JOBS`: Number of CPU cores for parallel execution.
- `N_PRECISION_QUBITS`: Defines the number of Grover iterations ($2^m$). 
- `n_estimators`: The number of trees in the forest. 
