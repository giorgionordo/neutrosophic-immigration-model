from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nsimmigration import ModelConfig, NeutrosophicImmigrationModel
from nsimmigration.plotting import (
    ensure_dir,
    save_integrator_evolution,
    save_score_distribution,
    save_snapshot_comparison,
    save_tipping_plot,
    save_topology_metrics,
    save_topology_structure,
)

THETA_I = 0.80
THETA_G = 0.25


def run_topology_experiment(figures_output: Path, results_output: Path):
    histories: dict[str, dict[str, list[float]]] = {}
    final_models: dict[str, NeutrosophicImmigrationModel] = {}

    for label, topology in [("ER", "er"), ("SW", "sw"), ("BA", "ba")]:
        cfg = ModelConfig(topology=topology, seed=42, n_steps=100, record_full_centrality=True)
        model = NeutrosophicImmigrationModel(cfg)
        histories[label] = model.run()
        final_models[label] = model

    save_topology_metrics(histories, figures_output / "fig_topology_metrics.png")
    save_integrator_evolution(histories, figures_output / "fig_integrator_evolution.png")
    save_topology_structure(histories, figures_output / "fig_topology_structure.png")
    save_snapshot_comparison(final_models, figures_output / "fig_network_snapshots.png")
    save_score_distribution(final_models, figures_output / "fig_score_distribution.png")

    rows = []
    for label, model in final_models.items():
        h = histories[label]
        rows.append(
            {
                "Scenario": label,
                "I_int": h["I_int"][-1],
                "v_out": h["v_out"][-1],
                "Integrator": h["top_integrator_mean"][-1],
                "Clustering": h["clustering"][-1],
                "Avg_path": h["avg_path"][-1],
                "Score_gap": h["score_gap"][-1],
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(results_output / "table_summary.csv", index=False)

    ba = final_models["BA"]
    centralities = ba.centralities()
    top_nodes = sorted(centralities.items(), key=lambda item: item[1]["integrator"], reverse=True)[:8]
    crows = []
    for node, d in top_nodes:
        group = "host" if ba.group(node) == 0 else f"guest-{ba.group(node)}"
        crows.append(
            {
                "Node": node,
                "Group": group,
                "Degree": int(d["degree"]),
                "Score": d["score"],
                "NDC": d["NDC"],
                "NCC": d["NCC"],
                "Betweenness": d["betweenness"],
                "Cross_share": d["cross_share"],
                "Integrator": d["integrator"],
            }
        )
    centrality_table = pd.DataFrame(crows)
    centrality_table.to_csv(results_output / "table_ba_centrality.csv", index=False)

    return histories, final_models, summary, centrality_table


def run_tipping_experiment(figures_output: Path, results_output: Path):
    fractions = [0.10, 0.18, 0.25, 0.33, 0.40, 0.48, 0.56, 0.62, 0.70]
    final_integration: list[float] = []
    final_gaps: list[float] = []

    for k, fraction in enumerate(fractions):
        cfg = ModelConfig(
            topology="sw",
            migrant_fraction=fraction,
            seed=100,
            n_steps=100,
            record_full_centrality=False,
        )
        model = NeutrosophicImmigrationModel(cfg)
        history = model.run()
        final_integration.append(history["I_int"][-1])
        final_gaps.append(history["score_gap"][-1])

    save_tipping_plot(
        fractions,
        final_integration,
        final_gaps,
        figures_output / "fig_tipping.png",
        theta_I=THETA_I,
        theta_g=THETA_G,
    )

    tipping = pd.DataFrame(
        {
            "migrant_fraction": fractions,
            "final_I_int": final_integration,
            "final_score_gap": final_gaps,
            "warning": [ii < THETA_I and gg > THETA_G for ii, gg in zip(final_integration, final_gaps)],
        }
    )
    tipping.to_csv(results_output / "table_tipping_scan.csv", index=False)
    return tipping


def save_config(results_output: Path) -> None:
    cfg = ModelConfig()
    data = {
        "N": cfg.n_agents,
        "n_migrant_groups": cfg.n_migrant_groups,
        "migrant_fraction": cfg.migrant_fraction,
        "T_max": cfg.n_steps,
        "target_average_degree": 6,
        "er_p": cfg.er_p,
        "ws_k": cfg.ws_k,
        "ws_rewire": cfg.ws_rewire,
        "ba_m": cfg.ba_m,
        "ba_seed_size": cfg.ba_seed_size,
        "chi": cfg.attractiveness,
        "lambda": cfg.lam,
        "sigma": cfg.sigma,
        "kappa_0": cfg.kappa0,
        "rho": cfg.density_rho,
        "beta": cfg.prestige_beta,
        "capacity_k0": cfg.capacity_k0,
        "capacity_a": cfg.capacity_a,
        "capacity_b": cfg.capacity_b,
        "capacity_c": cfg.capacity_c,
        "gamma": cfg.overload_gamma,
        "eta": cfg.preferential_eta,
        "learning_rate_alpha": cfg.learning_rate,
        "discount_Gamma": cfg.discount,
        "epsilon_greedy": cfg.exploration,
        "integration_bonus_zeta": cfg.integration_bonus,
        "distance_penalty_xi": cfg.distance_penalty,
        "distance_threshold_theta": cfg.distance_threshold,
        "neutral_trust_tau0": cfg.neutral_trust,
        "score_bins": cfg.score_bins,
        "load_bins": cfg.load_bins,
        "cross_bins": cfg.cross_bins,
        "integrator_weights": cfg.integrator_weights,
        "reward_matrix": NeutrosophicImmigrationModel(ModelConfig(record_full_centrality=False)).reward_matrix().tolist(),
        "tipping_theta_I": THETA_I,
        "tipping_theta_g": THETA_G,
        "directed_strength_initial_range": [0.55, 1.0],
        "rng_streams": "seed, seed+10000, seed+20000, seed+30000 for graph/attributes/edge strengths/dynamics",
    }
    with open(results_output / "reproducibility_config.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate manuscript figures, numerical tables, and reproducibility metadata."
    )
    parser.add_argument(
        "--figures-dir",
        "--output",
        dest="figures_dir",
        type=Path,
        default=ROOT / "figures",
        help="Directory for PNG figures (default: <repository>/figures).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "results",
        help="Directory for CSV/JSON numerical outputs (default: <repository>/results).",
    )
    args = parser.parse_args()

    figures_output = ensure_dir(args.figures_dir)
    results_output = ensure_dir(args.results_dir)

    histories, models, summary, centrality = run_topology_experiment(
        figures_output, results_output
    )
    tipping = run_tipping_experiment(figures_output, results_output)
    save_config(results_output)

    print("\nFinal topology summary:\n")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nTop BA cultural brokers:\n")
    print(centrality.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nTipping scan:\n")
    print(tipping.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nFigures saved in: {figures_output}")
    print(f"Results saved in: {results_output}")


if __name__ == "__main__":
    main()
