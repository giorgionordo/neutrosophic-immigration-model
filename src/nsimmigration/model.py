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
"""Agent-based SVNS immigration model with adaptive network dynamics."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Literal

import networkx as nx
import numpy as np

from .svns import Triplet, gaussian_similarity, positive_activation, project_triplet, score_distance, score_lambda

Topology = Literal["er", "sw", "ba"]
Action = Literal["keep", "add", "delete"]
ACTIONS: tuple[Action, ...] = ("keep", "add", "delete")


@dataclass
class ModelConfig:
    """Configuration of the SVNS immigration model."""

    n_agents: int = 90
    n_migrant_groups: int = 2
    migrant_fraction: float = 0.30
    topology: Topology = "sw"
    seed: int = 42
    n_steps: int = 100

    er_p: float = 0.07
    ws_k: int = 6
    ws_rewire: float = 0.12
    ba_m: int = 3
    attractiveness: float = 1.0

    lam: float = 0.60
    sigma: float = 0.55
    kappa0: float = 0.025
    density_rho: float = 0.30
    prestige_beta: float = 0.55
    capacity_k0: float = 6.0
    capacity_a: float = 2.0
    capacity_b: float = 2.0
    capacity_c: float = 2.0
    overload_gamma: float = 1.5
    epsilon: float = 1.0e-9

    preferential_eta: float = 0.45
    learning_rate: float = 0.18
    discount: float = 0.82
    exploration: float = 0.12
    integration_bonus: float = 0.10
    distance_penalty: float = 0.08
    distance_threshold: float = 0.75
    candidate_pairs_per_step: int = 60

    integrator_weights: tuple[float, float, float, float, float] = (0.18, 0.18, 0.22, 0.27, 0.15)
    reward_matrix: np.ndarray | None = None


@dataclass
class NeutrosophicImmigrationModel:
    """Simulation engine for the SVNS immigration model."""

    config: ModelConfig
    G: nx.Graph = field(init=False)
    rng: np.random.Generator = field(init=False)
    q_tables: Dict[int, Dict[tuple, Dict[Action, float]]] = field(init=False)
    previous_utility: Dict[int, float] = field(init=False)
    history: Dict[str, list[float]] = field(init=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.config.seed)
        self.G = self._initialize_graph()
        self.q_tables = {i: defaultdict(lambda: {a: 0.0 for a in ACTIONS}) for i in self.G.nodes}
        self.previous_utility = {i: 0.0 for i in self.G.nodes}
        self.history = defaultdict(list)
        self._record_metrics()

    def _initialize_graph(self) -> nx.Graph:
        c = self.config
        if c.topology == "er":
            G = nx.erdos_renyi_graph(c.n_agents, c.er_p, seed=c.seed)
        elif c.topology == "sw":
            k = min(c.ws_k + (c.ws_k % 2), c.n_agents - 1)
            G = nx.watts_strogatz_graph(c.n_agents, k, c.ws_rewire, seed=c.seed)
        elif c.topology == "ba":
            G = nx.barabasi_albert_graph(c.n_agents, c.ba_m, seed=c.seed)
        else:
            raise ValueError(f"Unknown topology: {c.topology}")

        n_migrants = int(round(c.n_agents * c.migrant_fraction))
        migrant_nodes = list(self.rng.choice(list(G.nodes), size=n_migrants, replace=False))
        migrant_index = {node: idx for idx, node in enumerate(migrant_nodes)}

        for i in G.nodes:
            if i in migrant_index:
                group = 1 + (migrant_index[i] % c.n_migrant_groups)
            else:
                group = 0
            G.nodes[i]["group"] = group
            G.nodes[i]["attitude"] = self._sample_attitude(group)

        for u, v in G.edges:
            self._initialize_edge_strength(G, u, v)
        return G

    def _initialize_edge_strength(self, G: nx.Graph, u: int, v: int) -> None:
        G.edges[u, v]["w_uv"] = float(self.rng.uniform(0.55, 1.0))
        G.edges[u, v]["w_vu"] = float(self.rng.uniform(0.55, 1.0))

    def _sample_attitude(self, group: int) -> Triplet:
        if group == 0:
            return project_triplet((self.rng.uniform(0.62, 0.94), self.rng.uniform(0.05, 0.30), self.rng.uniform(0.05, 0.28)))
        return project_triplet((self.rng.uniform(0.15, 0.48), self.rng.uniform(0.38, 0.78), self.rng.uniform(0.35, 0.78)))

    def group(self, i: int) -> int:
        return int(self.G.nodes[i]["group"])

    def attitude(self, i: int) -> Triplet:
        return self.G.nodes[i]["attitude"]

    def score(self, i: int) -> float:
        return score_lambda(self.attitude(i), self.config.lam)

    def similarity(self, i: int, j: int) -> float:
        return gaussian_similarity(self.attitude(i), self.attitude(j), self.config.sigma, self.config.lam)

    def reward_matrix(self) -> np.ndarray:
        c = self.config
        if c.reward_matrix is not None:
            return c.reward_matrix
        m = c.n_migrant_groups
        A = np.full((m + 1, m + 1), 0.72, dtype=float)
        A[0, 0] = 0.82
        for r in range(1, m + 1):
            A[r, r] = 0.78
            A[0, r] = 0.67
            A[r, 0] = 0.74
        return A

    def directed_strength(self, i: int, j: int) -> float:
        if not self.G.has_edge(i, j):
            return 0.0
        data = self.G.edges[i, j]
        return float(data.get("w_uv", 0.75) if i <= j else data.get("w_vu", 0.75))

    def degree_centrality(self, i: int) -> float:
        return self.G.degree(i) / max(1, self.config.n_agents - 1)

    def directed_reward(self, i: int, j: int) -> float:
        A = self.reward_matrix()
        beta = self.config.prestige_beta
        return float(A[self.group(i), self.group(j)] * self.similarity(i, j) * self.directed_strength(i, j) * (1.0 + beta * self.degree_centrality(j)))

    def aggregate_reward(self, i: int, j: int) -> float:
        return 0.5 * (self.directed_reward(i, j) + self.directed_reward(j, i))

    def capacity(self, i: int) -> float:
        c = self.config
        normalized_score = 0.5 * (self.score(i) + 1.0)
        return float(c.capacity_k0 * (1.0 + c.capacity_a * normalized_score + c.capacity_b * self.degree_centrality(i) + c.capacity_c * positive_activation(self.previous_utility[i])))

    def utility(self, i: int) -> float:
        c = self.config
        reward = sum(self.directed_reward(i, j) for j in self.G.neighbors(i))
        K = max(c.epsilon, self.capacity(i))
        overload = max(0.0, self.G.degree(i) - K) / K
        return float(reward - c.overload_gamma * overload * overload)

    def update_attitudes(self) -> None:
        c = self.config
        updates: dict[int, Triplet] = {}
        for i in self.G.nodes:
            neighbors = list(self.G.neighbors(i))
            if not neighbors:
                continue
            rewards = np.asarray([max(0.0, self.directed_reward(i, j)) for j in neighbors], dtype=float)
            if np.sum(rewards) <= 0.0:
                rewards = np.ones(len(neighbors), dtype=float)
            weights = rewards / (np.sum(rewards) + c.epsilon)
            drift = np.zeros(3, dtype=float)
            a_i = np.asarray(self.attitude(i), dtype=float)
            for w, j in zip(weights, neighbors):
                drift += w * (np.asarray(self.attitude(j), dtype=float) - a_i)
            kappa = min(1.0, c.kappa0 * (1.0 + c.density_rho * self.degree_centrality(i)))
            updates[i] = project_triplet(a_i + kappa * drift)
        for i, a in updates.items():
            self.G.nodes[i]["attitude"] = a

    def cross_share(self, i: int) -> float:
        neighbors = list(self.G.neighbors(i))
        if not neighbors:
            return 0.0
        return sum(1 for j in neighbors if self.group(j) != self.group(i)) / len(neighbors)

    def local_state(self, i: int) -> tuple[int, int, int]:
        score_bin = int(np.digitize(self.score(i), [-0.25, 0.25]))
        load = self.G.degree(i) / max(self.config.epsilon, self.capacity(i))
        load_bin = int(np.digitize(load, [0.75, 1.1]))
        cross_bin = int(np.digitize(self.cross_share(i), [0.25, 0.55]))
        return score_bin, load_bin, cross_bin

    def choose_action(self, i: int) -> Action:
        z = self.local_state(i)
        if self.rng.random() < self.config.exploration:
            return str(self.rng.choice(ACTIONS))  # type: ignore[return-value]
        q = self.q_tables[i][z]
        return max(ACTIONS, key=lambda a: q[a])

    def trust_score(self, i: int, j: int) -> float:
        """Simplified DSmT-inspired trust score based on common neighbours."""
        common = set(self.G.neighbors(i)).intersection(self.G.neighbors(j))
        if not common:
            return 0.50 * self.similarity(i, j)
        reliable, unreliable, partial = [], [], []
        for k in common:
            s = self.similarity(k, j)
            reliable.append(s * self.directed_strength(k, j))
            unreliable.append(1.0 - s)
            partial.append(0.5 * abs(self.score(k) - self.score(j)))
        m_r = float(np.mean(reliable))
        m_u = float(np.mean(unreliable))
        m_ri = float(max(0.0, 1.0 - np.mean(partial)) * (1.0 - abs(m_r - m_u)))
        return float(np.clip(m_r + 0.5 * m_ri - 0.5 * m_u, 0.0, 1.0))

    def add_probability(self, i: int, j: int) -> float:
        c = self.config
        if self.G.has_edge(i, j) or i == j:
            return 0.0
        z = self.local_state(i)
        raw = self.similarity(i, j) * ((self.G.degree(j) + c.attractiveness) ** c.preferential_eta)
        raw *= 1.0 + c.prestige_beta * self.degree_centrality(j)
        raw *= max(0.0, self.trust_score(i, j))
        raw *= float(np.exp(np.clip(self.q_tables[i][z]["add"], -6, 6)))
        return float(min(1.0, 0.025 * raw))

    def delete_probability(self, i: int, j: int) -> float:
        c = self.config
        if not self.G.has_edge(i, j):
            return 0.0
        max_reward = max([self.aggregate_reward(u, v) for u, v in self.G.edges] or [1.0])
        normalized_reward = self.aggregate_reward(i, j) / (max_reward + c.epsilon)
        z = self.local_state(i)
        logistic = 1.0 / (1.0 + np.exp(-np.clip(self.q_tables[i][z]["delete"], -6, 6)))
        overload = min(1.0, self.G.degree(i) / (self.capacity(i) + 1.0))
        return float(np.clip((1.0 - normalized_reward) * (1.0 - self.trust_score(i, j)) * overload * logistic, 0.0, 1.0))

    def rewire(self) -> None:
        nodes = np.array(list(self.G.nodes))
        for _ in range(self.config.candidate_pairs_per_step):
            i, j = map(int, self.rng.choice(nodes, size=2, replace=False))
            action = self.choose_action(i)
            old_state = self.local_state(i)
            old_utility = self.utility(i)
            old_I = self.integration_index()
            if action == "add" and not self.G.has_edge(i, j):
                if self.rng.random() < self.add_probability(i, j):
                    self.G.add_edge(i, j)
                    self._initialize_edge_strength(self.G, i, j)
            elif action == "delete" and self.G.has_edge(i, j):
                if self.rng.random() < self.delete_probability(i, j):
                    self.G.remove_edge(i, j)
            new_utility = self.utility(i)
            new_I = self.integration_index()
            dist_penalty = 0.0 if action == "keep" else float(score_distance(self.attitude(i), self.attitude(j), self.config.lam) > self.config.distance_threshold)
            reward_signal = (new_utility - old_utility) + self.config.integration_bonus * (new_I - old_I) - self.config.distance_penalty * dist_penalty
            self._update_q(i, old_state, action, reward_signal)

    def _update_q(self, i: int, z: tuple[int, int, int], action: Action, reward: float) -> None:
        c = self.config
        next_z = self.local_state(i)
        old = self.q_tables[i][z][action]
        future = max(self.q_tables[i][next_z].values())
        self.q_tables[i][z][action] = (1.0 - c.learning_rate) * old + c.learning_rate * (reward + c.discount * future)

    def step(self) -> None:
        self.previous_utility = {i: self.utility(i) for i in self.G.nodes}
        self.update_attitudes()
        self.rewire()
        self._record_metrics()

    def run(self) -> Dict[str, list[float]]:
        for _ in range(self.config.n_steps):
            self.step()
        return dict(self.history)

    def migrant_nodes(self) -> list[int]:
        return [i for i in self.G.nodes if self.group(i) != 0]

    def host_nodes(self) -> list[int]:
        return [i for i in self.G.nodes if self.group(i) == 0]

    def integration_index(self) -> float:
        c = self.config
        hosts = set(self.host_nodes())
        migrants = self.migrant_nodes()
        if not migrants or not hosts:
            return 0.0
        total = 0.0
        for i in migrants:
            d = self.G.degree(i)
            out = sum(1 for j in self.G.neighbors(i) if j in hosts)
            total += out / (d + c.epsilon)
        return float((c.n_agents / len(hosts)) * (total / len(migrants)))

    def v_out(self) -> float:
        c = self.config
        hosts = set(self.host_nodes())
        migrants = set(self.migrant_nodes())
        total_reward = sum(self.aggregate_reward(u, v) for u, v in self.G.edges)
        if total_reward <= c.epsilon or not hosts or not migrants:
            return 0.0
        cross_reward = 0.0
        for u, v in self.G.edges:
            if (u in hosts and v in migrants) or (v in hosts and u in migrants):
                cross_reward += self.aggregate_reward(u, v)
        normalization = c.n_agents * (c.n_agents - 1) / (2 * len(hosts) * len(migrants))
        return float((cross_reward / total_reward) * normalization)

    def score_gap(self) -> float:
        host_scores = [self.score(i) for i in self.host_nodes()]
        migrant_scores = [self.score(i) for i in self.migrant_nodes()]
        return float(abs(np.mean(host_scores) - np.mean(migrant_scores))) if host_scores and migrant_scores else 0.0

    def centralities(self) -> dict[int, dict[str, float]]:
        n = max(1, self.config.n_agents - 1)
        bc = nx.betweenness_centrality(self.G, normalized=True) if self.G.number_of_edges() else {i: 0.0 for i in self.G.nodes}
        out: dict[int, dict[str, float]] = {}
        for i in self.G.nodes:
            ndc = sum(self.similarity(i, j) * 0.5 * (self.directed_strength(i, j) + self.directed_strength(j, i)) for j in self.G.neighbors(i)) / n
            lengths = nx.single_source_shortest_path_length(self.G, i)
            ncc = sum((self.similarity(i, j) + self.config.epsilon) / ell for j, ell in lengths.items() if j != i) / n if len(lengths) > 1 else 0.0
            q = self.cross_share(i)
            score_rescaled = 0.5 * (self.score(i) + 1.0)
            w = self.config.integrator_weights
            B = w[0] * ndc + w[1] * ncc + w[2] * bc.get(i, 0.0) + w[3] * q + w[4] * score_rescaled
            out[i] = {"NDC": float(ndc), "NCC": float(ncc), "betweenness": float(bc.get(i, 0.0)), "cross_share": float(q), "score": self.score(i), "degree": float(self.G.degree(i)), "integrator": float(B)}
        return out

    def _fast_integrator_mean(self) -> float:
        # Fast proxy for time series: full betweenness is expensive and is computed only for final tables.
        vals = []
        for i in self.G.nodes:
            vals.append(0.35 * self.degree_centrality(i) + 0.45 * self.cross_share(i) + 0.20 * (0.5 * (self.score(i) + 1.0)))
        return float(np.mean(sorted(vals, reverse=True)[:10])) if vals else 0.0

    def _record_metrics(self) -> None:
        self.history["I_int"].append(self.integration_index())
        self.history["v_out"].append(self.v_out())
        self.history["score_gap"].append(self.score_gap())
        self.history["clustering"].append(float(nx.average_clustering(self.G)) if self.G.number_of_nodes() else 0.0)
        if self.G.number_of_edges() and self.G.number_of_nodes() > 1:
            largest = self.G.subgraph(max(nx.connected_components(self.G), key=len)).copy()
            path = nx.average_shortest_path_length(largest) if largest.number_of_nodes() > 1 else 0.0
        else:
            path = 0.0
        self.history["avg_path"].append(float(path))
        self.history["top_integrator_mean"].append(self._fast_integrator_mean())
