# Neutrosophic Immigration Model

Python implementation of the model described in the paper **A Neutrosophic Agent-Based Network Model for Immigration and Coexistence** by Giorgio Nordo, Carmelo Filippo Munafò, and Nivetha Martin.

Suggested GitHub repository name: `neutrosophic-immigration-model`.

## Features

- Single-valued neutrosophic attitudes `x_i(t)=<T_i(t), I_i(t), F_i(t)>`.
- Erdős-Rényi, Watts-Strogatz small-world, and Barabási-Albert scale-free initial networks.
- Capacity-dependent utility with heterogeneous group rewards.
- Reward-weighted SVNS attitude update.
- Simplified multi-agent Q-learning for adaptive add/keep/delete decisions.
- DSmT-inspired trust aggregation from common neighbours.
- Integration index, cross-group reward share, score gap, clustering, average path length, and cultural-broker / integrator index.
- Scripts for reproducing the manuscript figures.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run the experiments

```bash
python scripts/run_experiments.py --output figures
python scripts/generate_baseline_figures.py --output figures
```

The figures are saved as PNG files in the selected output folder.

## Repository creation

The GitHub connector available in ChatGPT can write to existing repositories but does not expose a `create repository` action. Create an empty repository named `neutrosophic-immigration-model` on GitHub and then push this folder:

```bash
git init
git add .
git commit -m "Initial implementation of the neutrosophic immigration model"
git branch -M main
git remote add origin git@github.com:giorgionordo/neutrosophic-immigration-model.git
git push -u origin main
```

## Citation

Please cite the related paper if this implementation is used in publications.
