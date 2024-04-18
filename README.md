# Molecular Energy Simulation using Monte Carlo Method

## Requirements:

- Python 3.11.5
- NumPy 1.26.4

## File Descriptions:

- `main.py`: The main script containing the implementation of the Monte Carlo simulation, including functions for calculating spin vectors, energy, and magnetization.
- `data.dat`: Output file containing simulation results.

## Usage:

- Ensure Python and NumPy are installed.
- Run `main.py`.
- Simulation results will be saved in the `data.dat` file.

## Methodology:

- **Spin Vector Generation:** The `get_spin_vecs` function generates spin vectors for the simulation.
- **Monte Carlo Simulation:** The `monte` function implements the Monte Carlo method for simulating molecular energy. It iterates over each lattice site and calculates the local energy, then updates the spin vectors based on energy changes.
- **Energy Calculation:** The `hs_energy` function calculates the total energy of the system based on spin configurations and interaction energies.
- **Magnetization Calculation:** The `magnetisation` function calculates the magnetization of the system based on spin configurations.

## Authors:

- This project was authored by Dillon Broaders.
