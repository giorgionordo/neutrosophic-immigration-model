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
"""Plotting utilities for the SVNS immigration model."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .model import NeutrosophicImmigrationModel


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_topology_metrics(histories: Dict[str, Dict[str, list[float]]], out_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=220)
    for label, h in histories.items():
        ax.plot(h["I_int"], label=f"{label}: integration index")
        ax.plot(h["v_out"], linestyle="--", label=f"{label}: cross-reward share")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Value")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Integration metrics across network topologies")
    ax.grid(True, alpha=0.30)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_integrator_evolution(histories: Dict[str, Dict[str, list[float]]], out_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=220)
    for label, h in histories.items():
        ax.plot(h["top_integrator_mean"], label=label)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Mean top-10 integrator index")
    ax.set_title("Evolution of cultural-broker potential")
    ax.grid(True, alpha=0.30)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_topology_structure(histories: Dict[str, Dict[str, list[float]]], out_path: str | Path) -> None:
    preferred_order = ["ER", "SW", "BA"]
    labels = [lab for lab in preferred_order if lab in histories]
    labels += [lab for lab in histories.keys() if lab not in labels]

    clustering = [histories[k]["clustering"][-1] for k in labels]
    paths = [histories[k]["avg_path"][-1] for k in labels]

    x = np.arange(len(labels))
    width = 0.36

    fig, ax1 = plt.subplots(figsize=(7.8, 5.1), dpi=220)
    ax2 = ax1.twinx()

    ax1.bar(
        x - width / 2,
        clustering,
        width,
        label="Clustering"
    )

    ax2.bar(
        x + width / 2,
        paths,
        width,
        alpha=0.60,
        label="Average path"
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Average clustering")
    ax2.set_ylabel("Average path length")

    if clustering:
        ax1.set_ylim(0.0, max(clustering) * 1.35)
    if paths:
        ax2.set_ylim(0.0, max(paths) * 1.25)

    fig.suptitle(
        "Structural signatures of the final networks",
        y=0.98
    )

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    fig.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.89),
        ncol=2,
        frameon=True,
        fontsize=8
    )
    ax1.grid(True, axis="y", alpha=0.25)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.86])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_tipping_plot(fractions: list[float], final_integration: list[float], gaps: list[float], out_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.8), dpi=220)
    ax.plot(fractions, final_integration, marker="o", label="Final integration index")
    ax.plot(fractions, gaps, marker="s", label="Final host-migrant score gap")
    ax.axhline(0.25, linestyle=":", label="segregation warning level")
    ax.set_xlabel("Migrant fraction")
    ax.set_ylabel("Value")
    ax.set_title("Neighbourhood tipping experiment (small-world topology)")
    ax.grid(True, alpha=0.30)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_snapshot_comparison(models: Dict[str, NeutrosophicImmigrationModel], out_path: str | Path) -> None:
    fig, axs = plt.subplots(1, len(models), figsize=(10.5, 4.5), dpi=220)
    if len(models) == 1:
        axs = [axs]
    nodes_artist = None
    for ax, (label, model) in zip(axs, models.items()):
        G = model.G
        pos = nx.spring_layout(G, seed=model.config.seed)
        node_colors = [model.score(i) for i in G.nodes]
        node_sizes = [35 + 8 * G.degree(i) for i in G.nodes]
        nx.draw_networkx_edges(G, pos=pos, ax=ax, edge_color="0.70", alpha=0.30, width=0.60)
        nodes_artist = nx.draw_networkx_nodes(G, pos=pos, ax=ax, node_color=node_colors, node_size=node_sizes, cmap="coolwarm", vmin=-1, vmax=1, linewidths=0)
        ax.set_title(label)
        ax.set_axis_off()
    if nodes_artist is not None:
        cbar = fig.colorbar(nodes_artist, ax=axs, shrink=0.75)
        cbar.set_label(r"Neutrosophic score $\mathrm{Sc}_\lambda$")
    fig.suptitle("Final network snapshots: color = score, size = degree", y=0.98)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_score_distribution(models: Dict[str, NeutrosophicImmigrationModel], out_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=220)
    bins = np.linspace(-1.0, 1.0, 28)
    for label, model in models.items():
        scores = [model.score(i) for i in model.G.nodes]
        ax.hist(scores, bins=bins, alpha=0.45, label=label)
    ax.set_xlabel(r"$\mathrm{Sc}_\lambda$")
    ax.set_ylabel("Frequency")
    ax.set_title("Final distribution of neutrosophic scores")
    ax.grid(True, alpha=0.30)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
