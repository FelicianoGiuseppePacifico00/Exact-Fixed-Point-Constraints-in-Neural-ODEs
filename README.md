# Exact Fixed-Point Constraints in Neural ODEs

This repository contains the code used to reproduce the numerical experiments in the paper.

## Contents

- `4_punti_fissi.ipynb`: experiment with four prescribed fixed points.
- `ciclo_limite.ipynb`: experiment with the limit-cycle system and one prescribed fixed point.
- `library9.py`: implementation of the constrained Neural ODE architecture, including the QR-based fixed-point planting construction.

## Requirements

The experiments require Python 3, PyTorch, NumPy, Matplotlib, and pandas.

## Reproducing the experiments

Run the notebooks in the following order:

1. Open `4_punti_fissi.ipynb` to reproduce the four-fixed-point experiment.
2. Open `ciclo_limite.ipynb` to reproduce the limit-cycle experiment.

Each notebook defines the target vector field, prescribed fixed points, model architecture, training procedure, evaluation grid, and plotting code. The multi-seed cells report mean and standard deviation across random seeds, as well as wall-clock time and compute information.

## Experimental setting

The experiments use analytically defined synthetic vector fields. No external datasets, pretrained models, human-subject data, or private data are used.

## Intended use

The code is released for research reproducibility. The method has not been validated for safety-critical deployment or real-world decision-making systems.
