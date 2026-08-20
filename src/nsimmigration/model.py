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

_R = frozenset({"R"})
_U = frozenset({"U"})
_RI = frozenset({"R", "I"})


@dataclass
class ModelConfig:
    """Configuration of the SVNS immigration model.

    Defaults match the numerical experiment described in the manuscript.
    """

    n_agents: int = 90
    n_migrant_groups: int = 2
    migrant_fraction: float = 0.30
    topology: Topology = "sw"
    seed: int = 42
    n_steps: int = 100

    # Initial topology.  For N=90, er_p=6/(N-1) gives target mean degree 6.
    er_p: float = 6.0 / 89.0
    ws_k: int = 6
    ws_rewire: float = 0.12
    ba_m: int = 3
    ba_seed_size: int = 4
    attractiveness: float = 1.0  # chi in generalized preferential attachment

    # SVNS / reward / utility parameters.
    lam: float = 0.60
    sigma: float = 0.55
    kappa0: float = 0.025
    density_rho: float = 0.30
    prestige_beta: float = 0.55
    capacity_k0: float = 6.0
    capacity_a: float = 0.05
    capacity_b: float = 0.05
    capacity_c: float = 0.05
    overload_gamma: float = 10.0
    epsilon: float = 1.0e-9

    # Learning / rewiring parameters.
    preferential_eta: float = 0.45
    learning_rate: float = 0.18
    discount: float = 0.82
    exploration: float = 0.12
    integration_bonus: float = 0.10
    distance_penalty: float = 0.08
    distance_threshold: float = 0.75
    neutral_trust: float = 0.50

    # Bins used in z_i^t=(bin(score),bin(load),bin(cross-share)).
    score_bins: tuple[float, float] = (-0.25, 0.25)
    load_bins: tuple[float, float] = (0.75, 1.10)
    cross_bins: tuple[float, float] = (0.25, 0.55)

    # alpha_1,...,alpha_5 in the integrator index.
    integrator_weights: tuple[float, float, float, float, float] = (0.18, 0.18, 0.22, 0.27, 0.15)

    # Computing exact NCC and betweenness at every time step is more expensive.
    record_full_centrality: bool = True

    reward_matrix: np.ndarray | None = None


@dataclass
class NeutrosophicImmigrationModel:
    """Simulation engine implementing the equations stated in the manuscript."""

    config: ModelConfig
    G: nx.Graph = field(init=False)
    rng: np.random.Generator = field(init=False)
    _attr_rng: np.random.Generator = field(init=False, repr=False)
    _edge_rng: np.random.Generator = field(init=False, repr=False)
    _graph_rng: np.random.Generator = field(init=False, repr=False)
    q_tables: Dict[int, Dict[tuple, Dict[Action, float]]] = field(init=False)
    previous_utility: Dict[int, float] = field(init=False)
    history: Dict[str, list[float]] = field(init=False)

    def __post_init__(self) -> None:
        self._validate_config()
        # Separate deterministic RNG streams keep group/attitude initialization
        # comparable across ER/SW/BA when the same seed is used.
        self._graph_rng = np.random.default_rng(self.config.seed)
        self._attr_rng = np.random.default_rng(self.config.seed + 10_000)
        self._edge_rng = np.random.default_rng(self.config.seed + 20_000)
        self.rng = np.random.default_rng(self.config.seed + 30_000)

        self.G = self._initialize_graph()
        self.q_tables = {i: defaultdict(lambda: {a: 0.0 for a in ACTIONS}) for i in self.G.nodes}
        # U_i^{-1}=0, as stated in the manuscript.
        self.previous_utility = {i: 0.0 for i in self.G.nodes}
        self.history = defaultdict(list)
        self._record_metrics()

    def _validate_config(self) -> None:
        c = self.config
        if c.n_agents < 2:
            raise ValueError("n_agents must be at least 2")
        if c.n_migrant_groups < 1:
            raise ValueError("n_migrant_groups must be positive")
        if not 0.0 <= c.migrant_fraction <= 1.0:
            raise ValueError("migrant_fraction must belong to [0,1]")
        if c.sigma <= 0.0:
            raise ValueError("sigma must be positive")
        if c.attractiveness < 0.0:
            raise ValueError("attractiveness chi must be non-negative")
        if not np.isclose(sum(c.integrator_weights), 1.0):
            raise ValueError("integrator_weights must sum to 1")

    # ------------------------------------------------------------------
    # Initial network and attributes
    # ------------------------------------------------------------------
    def _initialize_graph(self) -> nx.Graph:
        c = self.config
        graph_seed = int(self._graph_rng.integers(0, 2**32 - 1))
        if c.topology == "er":
            G = nx.erdos_renyi_graph(c.n_agents, c.er_p, seed=graph_seed)
        elif c.topology == "sw":
            k = min(c.ws_k + (c.ws_k % 2), c.n_agents - 1)
            G = nx.watts_strogatz_graph(c.n_agents, k, c.ws_rewire, seed=graph_seed)
        elif c.topology == "ba":
            G = self._generalized_preferential_attachment_graph()
        else:
            raise ValueError(f"Unknown topology: {c.topology}")

        n_migrants = int(round(c.n_agents * c.migrant_fraction))
        migrant_nodes = list(self._attr_rng.choice(list(G.nodes), size=n_migrants, replace=False))
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

    def _generalized_preferential_attachment_graph(self) -> nx.Graph:
        """Generalized BA construction with attachment proportional to d_j+chi.

        The seed is a complete connected graph on ba_seed_size vertices.  Targets
        for each new vertex are drawn without replacement according to the
        manuscript's preferential-attachment probabilities.
        """
        c = self.config
        seed_size = max(c.ba_m + 1, c.ba_seed_size)
        seed_size = min(seed_size, c.n_agents)
        if c.ba_m < 1 or c.ba_m >= seed_size:
            raise ValueError("ba_m must satisfy 1 <= ba_m < ba_seed_size")

        G = nx.complete_graph(seed_size)
        for new_node in range(seed_size, c.n_agents):
            existing = np.asarray(list(G.nodes), dtype=int)
            weights = np.asarray([G.degree(v) + c.attractiveness for v in existing], dtype=float)
            if np.sum(weights) <= 0.0:
                probabilities = np.full(len(existing), 1.0 / len(existing))
            else:
                probabilities = weights / np.sum(weights)
            targets = self._graph_rng.choice(existing, size=c.ba_m, replace=False, p=probabilities)
            G.add_node(new_node)
            G.add_edges_from((new_node, int(v)) for v in targets)
        return G

    def _initialize_edge_strength(self, G: nx.Graph, u: int, v: int) -> None:
        lo, hi = sorted((int(u), int(v)))
        # w_uv always denotes lo -> hi, w_vu denotes hi -> lo.
        G.edges[u, v]["w_uv"] = float(self._edge_rng.uniform(0.55, 1.0))
        G.edges[u, v]["w_vu"] = float(self._edge_rng.uniform(0.55, 1.0))
        G.edges[u, v]["ordered_endpoints"] = (lo, hi)

    def _sample_attitude(self, group: int) -> Triplet:
        if group == 0:
            return project_triplet(
                (
                    self._attr_rng.uniform(0.62, 0.94),
                    self._attr_rng.uniform(0.05, 0.30),
                    self._attr_rng.uniform(0.05, 0.28),
                )
            )
        return project_triplet(
            (
                self._attr_rng.uniform(0.15, 0.48),
                self._attr_rng.uniform(0.38, 0.78),
                self._attr_rng.uniform(0.35, 0.78),
            )
        )

    # ------------------------------------------------------------------
    # Basic quantities
    # ------------------------------------------------------------------
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
            A = np.asarray(c.reward_matrix, dtype=float)
            expected = (c.n_migrant_groups + 1, c.n_migrant_groups + 1)
            if A.shape != expected:
                raise ValueError(f"reward_matrix must have shape {expected}")
            return A

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
        lo, hi = data.get("ordered_endpoints", tuple(sorted((int(i), int(j)))))
        if i == lo and j == hi:
            return float(data.get("w_uv", 0.75))
        return float(data.get("w_vu", 0.75))

    def degree_centrality(self, i: int) -> float:
        return self.G.degree(i) / max(1, self.config.n_agents - 1)

    def directed_reward(self, i: int, j: int) -> float:
        if not self.G.has_edge(i, j):
            return 0.0
        A = self.reward_matrix()
        beta = self.config.prestige_beta
        return float(
            A[self.group(i), self.group(j)]
            * self.similarity(i, j)
            * self.directed_strength(i, j)
            * (1.0 + beta * self.degree_centrality(j))
        )

    def aggregate_reward(self, i: int, j: int) -> float:
        return 0.5 * (self.directed_reward(i, j) + self.directed_reward(j, i))

    def capacity(self, i: int) -> float:
        c = self.config
        normalized_score = 0.5 * (self.score(i) + 1.0)
        return float(
            c.capacity_k0
            * (
                1.0
                + c.capacity_a * normalized_score
                + c.capacity_b * self.degree_centrality(i)
                + c.capacity_c * positive_activation(self.previous_utility[i])
            )
        )

    def utility(self, i: int) -> float:
        c = self.config
        reward = sum(self.directed_reward(i, j) for j in self.G.neighbors(i))
        K = max(c.epsilon, self.capacity(i))
        overload = max(0.0, self.G.degree(i) - K) / K
        return float(reward - c.overload_gamma * overload * overload)

    # ------------------------------------------------------------------
    # Attitude dynamics: (A3)--(A5)
    # ------------------------------------------------------------------
    def update_attitudes(self) -> None:
        c = self.config
        updates: dict[int, Triplet] = {}
        for i in self.G.nodes:
            neighbors = list(self.G.neighbors(i))
            if not neighbors:
                continue

            rewards = np.asarray([self.directed_reward(i, j) for j in neighbors], dtype=float)
            denominator = float(np.sum(rewards) + c.epsilon)
            weights = rewards / denominator

            drift = np.zeros(3, dtype=float)
            a_i = np.asarray(self.attitude(i), dtype=float)
            for w, j in zip(weights, neighbors):
                drift += w * (np.asarray(self.attitude(j), dtype=float) - a_i)

            kappa = c.kappa0 * (1.0 + c.density_rho * self.degree_centrality(i))
            updates[i] = project_triplet(a_i + kappa * drift)

        for i, a in updates.items():
            self.G.nodes[i]["attitude"] = a

    # ------------------------------------------------------------------
    # Q-learning state and action selection: (A6)--(A8)
    # ------------------------------------------------------------------
    def cross_share(self, i: int) -> float:
        neighbors = list(self.G.neighbors(i))
        if not neighbors:
            return 0.0
        return float(sum(1 for j in neighbors if self.group(j) != self.group(i)) / len(neighbors))

    def local_state(self, i: int) -> tuple[int, int, int]:
        c = self.config
        score_bin = int(np.digitize(self.score(i), c.score_bins))
        load = self.G.degree(i) / max(c.epsilon, self.capacity(i))
        load_bin = int(np.digitize(load, c.load_bins))
        cross_bin = int(np.digitize(self.cross_share(i), c.cross_bins))
        return score_bin, load_bin, cross_bin

    def choose_action(self, i: int) -> Action:
        z = self.local_state(i)
        if self.rng.random() < self.config.exploration:
            return str(self.rng.choice(ACTIONS))  # type: ignore[return-value]
        q = self.q_tables[i][z]
        qmax = max(q.values())
        best = [a for a in ACTIONS if np.isclose(q[a], qmax)]
        # Conservative tie-breaking: when all actions are equally valued,
        # preserve the current neighbourhood. This avoids an artificial burst
        # of link creation at Q=0 and makes the no-information policy neutral.
        if "keep" in best:
            return "keep"
        return str(self.rng.choice(best))  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # DSmT-inspired trust: explicit BBA + (A9)--(A11)
    # ------------------------------------------------------------------
    def trust_bba(self, k: int, j: int) -> dict[frozenset[str], float]:
        """Basic belief assignment supplied by common neighbour k about j.

        This closes a specification gap in the manuscript.  The focal elements are
        Reliable, Unreliable, and Reliable∩Indeterminate:

          m(R)   = S_kj w_kj
          m(U)   = (1-S_kj) w_kj
          m(R∩I) = 1-w_kj.

        They are non-negative and sum exactly to one.
        """
        if not self.G.has_edge(k, j):
            raise ValueError("k must be adjacent to j to provide neighbour evidence")
        s = float(self.similarity(k, j))
        w = float(np.clip(self.directed_strength(k, j), 0.0, 1.0))
        return {_R: s * w, _U: (1.0 - s) * w, _RI: 1.0 - w}

    @staticmethod
    def _combine_dsmt_bbas(bbas: list[dict[frozenset[str], float]]) -> dict[frozenset[str], float]:
        """Simplified conjunctive DSm rule in the free model.

        Canonical focal elements are represented by sets of elementary labels.  A
        conjunction therefore accumulates labels (set union in this representation).
        """
        combined: dict[frozenset[str], float] = {frozenset(): 1.0}
        for bba in bbas:
            updated: dict[frozenset[str], float] = defaultdict(float)
            for A, mA in combined.items():
                for B, mB in bba.items():
                    updated[A.union(B)] += mA * mB
            combined = dict(updated)
        return combined

    def trust_masses(self, i: int, j: int) -> dict[frozenset[str], float]:
        common = sorted(set(self.G.neighbors(i)).intersection(self.G.neighbors(j)))
        if not common:
            return {}
        return self._combine_dsmt_bbas([self.trust_bba(k, j) for k in common])

    def trust_score(self, i: int, j: int) -> float:
        common = set(self.G.neighbors(i)).intersection(self.G.neighbors(j))
        if not common:
            return float(np.clip(self.config.neutral_trust, 0.0, 1.0))

        masses = self.trust_masses(i, j)
        raw = masses.get(_R, 0.0) + 0.5 * masses.get(_RI, 0.0) - 0.5 * masses.get(_U, 0.0)
        return float(np.clip(raw, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Rewiring: (A12)--(A13)
    # ------------------------------------------------------------------
    def _add_weight(self, i: int, j: int) -> float:
        c = self.config
        if i == j or self.G.has_edge(i, j):
            return 0.0
        z = self.local_state(i)
        qfactor = float(np.exp(np.clip(self.q_tables[i][z]["add"], -20.0, 20.0)))
        return float(
            self.similarity(i, j)
            * ((self.G.degree(j) + c.attractiveness) ** c.preferential_eta)
            * (1.0 + c.prestige_beta * self.degree_centrality(j))
            * self.trust_score(i, j)
            * qfactor
        )

    def add_distribution(self, i: int) -> tuple[list[int], np.ndarray]:
        candidates = [j for j in self.G.nodes if j != i and not self.G.has_edge(i, j)]
        if not candidates:
            return [], np.asarray([], dtype=float)
        weights = np.asarray([self._add_weight(i, j) for j in candidates], dtype=float)
        Z = float(np.sum(weights))
        if Z <= self.config.epsilon:
            return [], np.asarray([], dtype=float)
        return candidates, weights / Z

    def add_probability(self, i: int, j: int) -> float:
        candidates, probs = self.add_distribution(i)
        try:
            idx = candidates.index(j)
        except ValueError:
            return 0.0
        return float(probs[idx])

    def choose_add_target(self, i: int) -> int | None:
        candidates, probs = self.add_distribution(i)
        if not candidates:
            return None
        return int(self.rng.choice(np.asarray(candidates, dtype=int), p=probs))

    def delete_probability(self, i: int, j: int) -> float:
        c = self.config
        if not self.G.has_edge(i, j):
            return 0.0

        rewards = [self.aggregate_reward(u, v) for u, v in self.G.edges]
        max_reward = max(rewards) if rewards else 0.0
        normalized_reward = self.aggregate_reward(i, j) / (max_reward + c.epsilon)

        z = self.local_state(i)
        q = float(np.clip(self.q_tables[i][z]["delete"], -20.0, 20.0))
        logistic = 1.0 / (1.0 + np.exp(-q))
        overload = min(1.0, self.G.degree(i) / (self.capacity(i) + 1.0))

        p = (1.0 - normalized_reward) * (1.0 - self.trust_score(i, j)) * overload * logistic
        return float(np.clip(p, 0.0, 1.0))

    def _update_q(self, i: int, z: tuple[int, int, int], action: Action, reward: float) -> None:
        c = self.config
        next_z = self.local_state(i)
        old = self.q_tables[i][z][action]
        future = max(self.q_tables[i][next_z].values())
        self.q_tables[i][z][action] = (1.0 - c.learning_rate) * old + c.learning_rate * (
            reward + c.discount * future
        )

    def rewire(self, utility_t: Dict[int, float]) -> None:
        """One sequential keep/add/delete decision for each agent.

        Add: choose one non-neighbour from the normalized distribution (A12).
        Delete: choose one current neighbour uniformly, then delete it with the
        conditional probability (A13).  The latter convention makes the singular
        'delete one existing link' instruction in the manuscript algorithm explicit.
        """
        c = self.config
        nodes = np.asarray(list(self.G.nodes), dtype=int)
        for i in self.rng.permutation(nodes):
            i = int(i)
            old_state = self.local_state(i)
            action = self.choose_action(i)
            old_I = self.integration_index()
            selected_j: int | None = None

            if action == "add":
                selected_j = self.choose_add_target(i)
                if selected_j is not None:
                    self.G.add_edge(i, selected_j)
                    self._initialize_edge_strength(self.G, i, selected_j)

            elif action == "delete":
                neighbors = list(self.G.neighbors(i))
                if neighbors:
                    selected_j = int(self.rng.choice(np.asarray(neighbors, dtype=int)))
                    if self.rng.random() < self.delete_probability(i, selected_j):
                        self.G.remove_edge(i, selected_j)

            new_utility = self.utility(i)
            new_I = self.integration_index()
            dist_penalty = 0.0
            if selected_j is not None and action != "keep":
                dist_penalty = float(
                    score_distance(self.attitude(i), self.attitude(selected_j), c.lam) > c.distance_threshold
                )

            reward_signal = (
                (new_utility - utility_t[i])
                + c.integration_bonus * (new_I - old_I)
                - c.distance_penalty * dist_penalty
            )
            self._update_q(i, old_state, action, reward_signal)

    # ------------------------------------------------------------------
    # Simulation clock
    # ------------------------------------------------------------------
    def step(self) -> None:
        # U_i^t is computed from x^t,G^t using U_i^{t-1} in K_i^t.
        utility_t = {i: self.utility(i) for i in self.G.nodes}

        # (A3): x^t -> x^{t+1} while G^t is still fixed.
        self.update_attitudes()

        # U_i^t now becomes the lagged utility used in K_i^{t+1}.
        self.previous_utility = utility_t

        # Sequential learning-based rewiring G^t -> G^{t+1}.
        self.rewire(utility_t)
        self._record_metrics()

    def run(self) -> Dict[str, list[float]]:
        for _ in range(self.config.n_steps):
            self.step()
        return dict(self.history)

    # ------------------------------------------------------------------
    # Macroscopic observables: (A1)--(A2)
    # ------------------------------------------------------------------
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
        if not hosts or not migrants:
            return 0.0
        total_reward = sum(self.aggregate_reward(u, v) for u, v in self.G.edges)
        if total_reward <= c.epsilon:
            return 0.0
        cross_reward = 0.0
        for u, v in self.G.edges:
            if (u in hosts and v in migrants) or (v in hosts and u in migrants):
                cross_reward += self.aggregate_reward(u, v)
        normalization = c.n_agents * (c.n_agents - 1) / (2.0 * len(hosts) * len(migrants))
        return float((cross_reward / (total_reward + c.epsilon)) * normalization)

    def score_gap(self) -> float:
        host_scores = [self.score(i) for i in self.host_nodes()]
        migrant_scores = [self.score(i) for i in self.migrant_nodes()]
        return float(abs(np.mean(host_scores) - np.mean(migrant_scores))) if host_scores and migrant_scores else 0.0

    # ------------------------------------------------------------------
    # Exact node-level observables from Section 5
    # ------------------------------------------------------------------
    def _ncc(self, i: int) -> float:
        """Neutrosophic closeness centrality exactly as defined in the manuscript."""
        if self.G.degree(i) == 0:
            return 0.0
        paths = nx.single_source_shortest_path(self.G, i)
        denominator = 0.0
        reachable = 0
        for j, path in paths.items():
            if j == i:
                continue
            ell = len(path) - 1
            if ell <= 0:
                continue
            similarities = [self.similarity(path[h], path[h + 1]) for h in range(ell)]
            avg_similarity = float(np.mean(similarities)) if similarities else 0.0
            denominator += ell / (avg_similarity + self.config.epsilon)
            reachable += 1
        if reachable == 0 or denominator <= 0.0:
            return 0.0
        return float(reachable / denominator)

    def centralities(self) -> dict[int, dict[str, float]]:
        n = max(1, self.config.n_agents - 1)
        if self.G.number_of_edges():
            bc = nx.betweenness_centrality(self.G, normalized=True)
        else:
            bc = {i: 0.0 for i in self.G.nodes}

        out: dict[int, dict[str, float]] = {}
        w = self.config.integrator_weights
        for i in self.G.nodes:
            ndc = (
                sum(
                    self.similarity(i, j)
                    * 0.5 * (self.directed_strength(i, j) + self.directed_strength(j, i))
                    for j in self.G.neighbors(i)
                )
                / n
            )
            ncc = self._ncc(i)
            q = self.cross_share(i)
            score_rescaled = 0.5 * (self.score(i) + 1.0)
            B = w[0] * ndc + w[1] * ncc + w[2] * bc.get(i, 0.0) + w[3] * q + w[4] * score_rescaled
            out[i] = {
                "NDC": float(ndc),
                "NCC": float(ncc),
                "betweenness": float(bc.get(i, 0.0)),
                "cross_share": float(q),
                "score": float(self.score(i)),
                "degree": float(self.G.degree(i)),
                "integrator": float(B),
            }
        return out

    def top_integrator_mean(self, k: int = 10) -> float:
        vals = sorted((d["integrator"] for d in self.centralities().values()), reverse=True)
        return float(np.mean(vals[:k])) if vals else 0.0

    # ------------------------------------------------------------------
    # Recorded diagnostics
    # ------------------------------------------------------------------
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

        if self.config.record_full_centrality:
            self.history["top_integrator_mean"].append(self.top_integrator_mean(10))
