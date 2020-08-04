# Coupled Mode Sampling
This program uses QChem frequency outputs to generate coupled mode potential energy surfaces by Monte Carlo sampling. The samples are drawn from the distribution from the sum of uncoupled mode approximation (UM-VT) potentials.

## Getting started
Clone methods into a source directory
`git clone https://github.com/lbettins/coupled_sampling`

Export this to your Python path. Ideally this can be added to `~/.bashrc` and sourced through `source ~/.bashrc` (or a similar `~/.*rc` file) as needed.
`

This code builds off of UM-VT sampling from the Yi-Pei Li group. See https://github.com/shihchengli/APE for more information.

## Prerequisites
Install Anaconda and create RMG-Py environment. Activate the environment.
`conda activate rmg\_env`
Download in the new environment:
RMG-Py
RDKit 
PyStan

Export RDKit directory to Python path.
