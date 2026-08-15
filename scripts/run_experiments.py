# ============================================================================
# Project: A Neutrosophic Agent-Based Network Model for Immigration and Coexistence
# Main author: Giorgio Nordo
# Affiliation: Department of Mathematical and Computer Sciences, Physical Sciences
#              and Earth Sciences (MIFT), University of Messina, Italy
# E-mail: giorgio.nordo@unime.it
# Website: https://www.nordo.it
# Coauthors of the related paper: Carmelo Filippo Munafò, Nivetha Martin
# Suggested repository: https://github.com/giorgionordo/neutrosophic-immigration-model
# ============================================================================
"""Run reproducible experiments and generate manuscript figures."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nsimmigration.model import ModelConfig, NeutrosophicImmigrationModel
from nsimmigration.plotting import (
    ensure_dir,
    save_integrator_evolution,
    save_score_distribution,
    save_snapshot_comparison,
    save_tipping_plot,
    save_topology_metrics,
    save_topology_structure,
)


def run_topology_experiment(output: Path) -> dict[str, dict[str, list[float]]]:
    histories: dict[str, dict[str, list[float]]] = {}
    final_models: dict[str, NeutrosophicImmigrationModel] = {}
    for label, topology in [("ER", "er"), ("SW", "sw"), ("BA", "ba")]:
        cfg = ModelConfig(topology=topology, seed=42, n_steps=80, candidate_pairs_per_step=45)
        model = NeutrosophicImmigrationModel(cfg)
        histories[label] = model.run()
        final_models[label] = model
    save_topology_metrics(histories, output / "fig_topology_metrics_generated.png")
    save_integrator_evolution(histories, output / "fig_integrator_evolution_generated.png")
    save_topology_structure(histories, output / "fig_topology_structure_generated.png")
    save_snapshot_comparison(final_models, output / "fig_network_snapshots.png")
    save_score_distribution(final_models, output / "fig_score_distribution.png")
    return histories


def run_tipping_experiment(output: Path) -> None:
    fractions = [0.10, 0.18, 0.25, 0.33, 0.40, 0.48, 0.56, 0.62, 0.70]
    final_integration: list[float] = []
    final_gaps: list[float] = []
    for k, fraction in enumerate(fractions):
        cfg = ModelConfig(topology="sw", migrant_fraction=fraction, seed=100 + k, n_steps=60, candidate_pairs_per_step=35)
        model = NeutrosophicImmigrationModel(cfg)
        history = model.run()
        final_integration.append(history["I_int"][-1])
        final_gaps.append(history["score_gap"][-1])
    save_tipping_plot(fractions, final_integration, final_gaps, output / "fig_tipping_generated.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate figures for the neutrosophic immigration model.")
    parser.add_argument("--output", type=Path, default=ROOT / "figures", help="Output folder for PNG figures.")
    args = parser.parse_args()
    output = ensure_dir(args.output)
    run_topology_experiment(output)
    run_tipping_experiment(output)
    print(f"Figures saved in: {output}")


if __name__ == "__main__":
    main()
