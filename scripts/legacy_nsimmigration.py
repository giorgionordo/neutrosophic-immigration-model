import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm

# ============================================================
#  Neutrosophic Immigration Model (SVNS-based)
#  - Attitude of node i: A_i = (T_i, I_i, F_i) in [0,1]^3
#  - Scalarization for decisions/plots: Score_lambda(A)
#  - Update uses weighted neighbor influence (neutrosophic vector update)
#  - Rewiring uses probability based on distance of scores (or full vector)
# ============================================================

# ============================================================
# Color constants (GLOBAL)
# ============================================================

# Structural figure (groups)
HOST_COLOR = "#2ca02c"  # green
GUEST_COLOR = "#9467bd"  # purple

# Cognitive figure (semantic reference)
NEGATIVE_SCORE_COLOR = "#1f77b4"  # blue
POSITIVE_SCORE_COLOR = "#d62728"  # red

# -----------------------
# Global parameters
# -----------------------
NUM_NODES = 200
NUM_GUESTS = 30
NUM_HOSTS = NUM_NODES - NUM_GUESTS
PROPORTION_GUESTS = NUM_GUESTS / NUM_NODES  # proporzione guest sul totale (solo descrittiva)

VERBOSE = False

TIMESTEPS = 2000

SIGMA = 0.5  # sensitivity in similarity/probability 0.5
KAPPA = 0.05  # cultural adaptation step 0.08
INITIAL_CONNECTION_PROB = 0.1

LAMBDA_SCORE = 0.6  # lambda in Score_lambda (0..1) 0.6
USE_VECTOR_DISTANCE = False  # if True: utility based on ||A_i - A_j||; else based on score distance

SEED = 42


# -----------------------
# Helpers
# -----------------------
def clamp01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def clamp_triplet(a):
    return (clamp01(a[0]), clamp01(a[1]), clamp01(a[2]))


def score_lambda(a, lam=LAMBDA_SCORE):
    """
    A = (T,I,F). A common scalar score used in SVNS decision settings.
    Higher -> more "positive" attitude.
    """
    T, I, F = a
    return T - lam * F - (1.0 - lam) * I


def neutro_distance(a, b, lam=LAMBDA_SCORE):
    """
    Distance used to compute similarity/utility.
    - If USE_VECTOR_DISTANCE: Euclidean in (T,I,F)
    - Else: absolute distance between scalar scores
    """
    if USE_VECTOR_DISTANCE:
        return float(np.linalg.norm(np.array(a) - np.array(b)))
    return abs(score_lambda(a, lam) - score_lambda(b, lam))


def utility(a_i, a_j, sigma=SIGMA, lam=LAMBDA_SCORE):
    """
    Similarity/utility in [0,1].
    Classical model uses exp(-(Δatt)^2 / (2 sigma^2)).
    Here Δatt is replaced by neutro_distance.
    """
    d = neutro_distance(a_i, a_j, lam)
    return float(np.exp(-(d ** 2) / (2.0 * sigma ** 2)))


# -----------------------
# Initialization
# -----------------------

def sample_attitude(node_type: str) -> tuple[float, float, float]:
    """
    Initialize SVNS attitude for host/guest with stronger separation,
    in the spirit of the MATLAB model (hosts more positive, guests more negative),
    while keeping the neutrosophic triplet (T,I,F) in [0,1]^3.
    """
    if node_type == "guest":
        # "More negative" on the scalar score:
        # lower T, higher I and F
        T = np.random.uniform(0.05, 0.35)
        I = np.random.uniform(0.55, 0.90)
        F = np.random.uniform(0.55, 0.90)
        return clamp_triplet((T, I, F))
    # host
    # "More positive" on the scalar score:
    # higher T, lower I and F
    T = np.random.uniform(0.65, 0.95)
    I = np.random.uniform(0.05, 0.30)
    F = np.random.uniform(0.05, 0.30)
    return clamp_triplet((T, I, F))


def initialize_network(num_nodes,
                       num_guests,
                       initial_connection_prob,
                       seed=SEED):
    np.random.seed(seed)
    G = nx.erdos_renyi_graph(num_nodes, initial_connection_prob, seed=seed)

    # scegli ESATTAMENTE num_guests nodi guest
    all_nodes = np.array(list(G.nodes()))
    guest_nodes = set(np.random.choice(all_nodes, size=num_guests, replace=False))

    for i in G.nodes():
        node_type = "guest" if i in guest_nodes else "host"
        G.nodes[i]["type"] = node_type
        G.nodes[i]["attitude"] = sample_attitude(node_type)
    return G


# -----------------------
# Attitude update (neutrosophic)
# -----------------------
def update_attitudes(G, kappa=KAPPA, lam=LAMBDA_SCORE):
    """
    Neutrosophic analogue of:
      a_i <- a_i + kappa * sum_j u(i,j) * (a_j - a_i)

    Here a_i is a 3-vector (T,I,F).
    """
    new_att = {}

    for i in G.nodes():
        a_i = G.nodes[i]["attitude"]
        neigh = list(G.neighbors(i))
        if not neigh:
            new_att[i] = a_i
            continue

        # Weighted drift toward neighbors
        drift = np.zeros(3, dtype=float)
        for j in neigh:
            a_j = G.nodes[j]["attitude"]
            w = utility(a_i, a_j, SIGMA, lam)
            drift += w * (np.array(a_j) - np.array(a_i))

        a_next = np.array(a_i) + kappa * drift

        # Optional: normalize to avoid T+I+F exploding (not required for SVNS),
        # but keeping each component in [0,1] is essential.
        new_att[i] = clamp_triplet(tuple(a_next))

    for i, a in new_att.items():
        G.nodes[i]["attitude"] = a


# -----------------------
# Rewiring
# -----------------------
def rewire_connections(G, lam=LAMBDA_SCORE):
    """
    Neutrosophic analogue of your rewiring:
      prob = utility(att_i, att_j)
      if prob > rand: add edge else remove

    Complexity note: O(N^2). For large N use sampling.
    """
    nodes = list(G.nodes())
    n = len(nodes)

    for idx_i in range(n):
        i = nodes[idx_i]
        a_i = G.nodes[i]["attitude"]
        for idx_j in range(idx_i + 1, n):
            j = nodes[idx_j]
            a_j = G.nodes[j]["attitude"]

            p = utility(a_i, a_j, SIGMA, lam)
            r = np.random.rand()

            if p > r:
                G.add_edge(i, j)
            else:
                if G.has_edge(i, j):
                    G.remove_edge(i, j)


# -----------------------
# Metrics (paper-friendly)
# -----------------------
def integration_index(G, lam=LAMBDA_SCORE):
    """
    One simple integration index: average fraction of cross-group edges.
    I_int = (# edges host-guest) / (# edges total)    in [0,1]
    """
    m = G.number_of_edges()
    if m == 0:
        return 0.0

    cross = 0
    for u, v in G.edges():
        if G.nodes[u]["type"] != G.nodes[v]["type"]:
            cross += 1
    return cross / m


def v_out_reward(G, lam=LAMBDA_SCORE):
    """
    Cross-group 'reward share' analogue:
    v_out = (sum_{host-guest edges} utility(i,j)) / (sum_{all edges} utility(i,j))
    """
    denom = 0.0
    numer = 0.0
    for u, v in G.edges():
        a_u = G.nodes[u]["attitude"]
        a_v = G.nodes[v]["attitude"]
        w = utility(a_u, a_v, SIGMA, lam)
        denom += w
        if G.nodes[u]["type"] != G.nodes[v]["type"]:
            numer += w
    return 0.0 if denom == 0.0 else numer / denom


# -----------------------
# Simulation
# -----------------------
def simulate_model(num_nodes=NUM_NODES,
                   num_guests=NUM_GUESTS,
                   timesteps=TIMESTEPS,
                   initial_connection_prob=INITIAL_CONNECTION_PROB,
                   lam=LAMBDA_SCORE,
                   seed=SEED):
    G = initialize_network(num_nodes, num_guests, initial_connection_prob, seed=seed)

    # quick sanity check (optional)
    host_scores = []
    guest_scores = []
    for i in G.nodes():
        s = score_lambda(G.nodes[i]["attitude"], LAMBDA_SCORE)
        if G.nodes[i]["type"] == "host":
            host_scores.append(s)
        else:
            guest_scores.append(s)

    if VERBOSE:
        print("Host score mean:", np.mean(host_scores),
              "min/max:", np.min(host_scores), np.max(host_scores))
        print("Guest score mean:", np.mean(guest_scores),
              "min/max:", np.min(guest_scores), np.max(guest_scores))

    # Store time series of scalar scores for plotting + metrics
    scores_over_time = []
    I_int_series = []
    v_out_series = []

    for t in range(timesteps):
        scores = [score_lambda(G.nodes[i]["attitude"], lam) for i in G.nodes()]
        scores_over_time.append(scores)

        I_int_series.append(integration_index(G, lam))
        v_out_series.append(v_out_reward(G, lam))

        update_attitudes(G, kappa=KAPPA, lam=lam)
        rewire_connections(G, lam=lam)

    return G, scores_over_time, I_int_series, v_out_series


def simulate_model_scenario(num_nodes=NUM_NODES,
                            num_guests=NUM_GUESTS,
                            timesteps=TIMESTEPS,
                            initial_connection_prob=INITIAL_CONNECTION_PROB,
                            lam=LAMBDA_SCORE,
                            seed=SEED,
                            do_rewire=True):
    """
    Like simulate_model(), but configurable:
      - do_rewire=True  -> rewiring ON (co-evolution of topology + attitudes)
      - do_rewire=False -> rewiring OFF (frozen topology; only attitudes evolve)

    Returns:
      G_final, scores_over_time (list[list[float]]), I_int_series, v_out_series
    """
    G = initialize_network(num_nodes, num_guests, initial_connection_prob, seed=seed)

    scores_over_time = []
    I_int_series = []
    v_out_series = []

    for t in range(timesteps):
        scores = [score_lambda(G.nodes[i]["attitude"], lam) for i in G.nodes()]
        scores_over_time.append(scores)

        I_int_series.append(integration_index(G, lam))
        v_out_series.append(v_out_reward(G, lam))

        update_attitudes(G, kappa=KAPPA, lam=lam)
        if do_rewire:
            rewire_connections(G, lam=lam)

    return G, scores_over_time, I_int_series, v_out_series


# ============================================================
# Paper-style graph snapshots (two-group layout, stable over time)
# ============================================================
def initial_pos_by_group(G, sep=3.0, jitter=0.35, seed=SEED):
    """Place hosts on the left blob and guests on the right blob."""
    rng = np.random.default_rng(seed)
    pos = {}
    for i in G.nodes():
        x0 = (-sep / 2.0) if G.nodes[i]["type"] == "host" else (sep / 2.0)
        pos[i] = (x0 + rng.normal(0, jitter), rng.normal(0, jitter))
    return pos


def stable_layout(G, pos0, it=10, k=None, seed=SEED):
    """
    Run a few spring-layout iterations starting from pos0, to avoid random
    rotations/flips and keep snapshots visually comparable across time.
    """
    return nx.spring_layout(G, pos=pos0, iterations=it, k=k, seed=seed)


def draw_paper_style(ax, G, pos, node_size=9):
    """Draw graph in 'paper style': gray translucent edges, blue hosts, red guests."""
    hosts = [i for i in G.nodes() if G.nodes[i]["type"] == "host"]
    guests = [i for i in G.nodes() if G.nodes[i]["type"] == "guest"]

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="0.7", alpha=0.25, width=0.5)
    nx.draw_networkx_nodes(G, pos, nodelist=hosts, ax=ax,
                           node_color="#1f77b4", node_size=node_size, linewidths=0)
    nx.draw_networkx_nodes(G, pos, nodelist=guests, ax=ax,
                           node_color="#d62728", node_size=node_size, linewidths=0)

    ax.set_axis_off()


def simulate_and_store_snapshots(
        timesteps,
        snapshot_ts,
        num_nodes=NUM_NODES,
        num_guests=NUM_GUESTS,
        initial_connection_prob=INITIAL_CONNECTION_PROB,
        lam=LAMBDA_SCORE,
        seed=SEED,
        layout_iterations=10,
        do_rewire=True,
):
    """
    Run the model and store (G_t, pos_t) at selected timesteps.
    NOTE: for very large timesteps with many nodes, rewiring is the bottleneck (O(N^2)).
    """
    G = initialize_network(num_nodes, num_guests, initial_connection_prob, seed=seed)
    pos = initial_pos_by_group(G, seed=seed)

    snaps = {}
    snapshot_ts = set(snapshot_ts)

    for t in range(timesteps + 1):
        if t in snapshot_ts:
            snaps[t] = (G.copy(), pos.copy())

        update_attitudes(G, kappa=KAPPA, lam=lam)
        if do_rewire:
            rewire_connections(G, lam=lam)

        # keep layout stable but responsive
        pos = stable_layout(G, pos, it=layout_iterations, seed=seed)

    return snaps


def plot_two_row_snapshots(snaps_a, snaps_b, ts, labels=("a", "b"), figsize=(12, 6), dpi=300):
    fig, axs = plt.subplots(2, 3, figsize=figsize, dpi=dpi)

    for r, (lab, snaps) in enumerate(zip(labels, [snaps_a, snaps_b])):
        for c, t in enumerate(ts):
            Gt, post = snaps[t]
            draw_paper_style(axs[r, c], Gt, post, node_size=8)

            if c == 0:
                axs[r, c].text(0.02, 0.98, f"({lab})", transform=axs[r, c].transAxes,
                               ha="left", va="top", fontsize=12)
            axs[r, c].set_title(f"$t={t}$", fontsize=11)

    plt.tight_layout()
    plt.show()


# ============================================================
# Run (paper-style snapshots)
# ============================================================

# (legacy snapshot block removed; snapshots are generated in the __main__ section below)

# ============================================================
# Optional diagnostics (histograms / metrics / score-colored graph)
# ============================================================
PLOT_DIAGNOSTICS = False

if PLOT_DIAGNOSTICS:
    G, scores_over_time, I_int_series, v_out_series = simulate_model()

    # Plot 1: score distribution over time (histograms)
    plt.figure(figsize=(10, 6))
    bins = np.linspace(-1, 1, 30)
    for t, scores in enumerate(scores_over_time):
        if t % 10 == 0 or t == TIMESTEPS - 1:
            plt.hist(scores, bins=bins, alpha=0.5, label=f'Timestep {t}')
    plt.title(r'Evoluzione degli score neutrosofici (Score$_\lambda$)')
    plt.xlabel(r'Score$_\lambda$')
    plt.ylabel('Frequenza')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot 2: final network colored by score
    plt.figure(figsize=(12, 12))
    node_scores = np.array([score_lambda(G.nodes[i]["attitude"], LAMBDA_SCORE) for i in G.nodes()])

    norm = Normalize(vmin=-1, vmax=1)
    cmap = cm.coolwarm
    colors = [cmap(norm(s)) for s in node_scores]

    fig, ax = plt.subplots(figsize=(12, 12))
    nx.draw_networkx(G, node_color=colors, with_labels=False, node_size=90, edge_color='gray', ax=ax)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label=r'Score$_\lambda$')
    plt.title(r'Rete finale colorata per Score$_\lambda$')
    plt.show()

    # Plot 3: degree distribution
    degree_sequence = [d for _, d in G.degree()]
    plt.figure(figsize=(10, 6))
    plt.hist(degree_sequence, bins=range(0, max(degree_sequence) + 2), alpha=0.75, edgecolor='black')
    plt.title('Distribuzione del grado nella rete finale')
    plt.xlabel('Grado')
    plt.ylabel('Frequenza')
    plt.grid(True)
    plt.show()

    # Plot 4: integration metrics over time
    plt.figure(figsize=(10, 6))
    plt.plot(I_int_series, label='I_int (cross-edge fraction)')
    plt.plot(v_out_series, label='v_out (cross reward share)')
    plt.title('Metriche di integrazione nel tempo')
    plt.xlabel('Timestep')
    plt.ylabel('Valore')
    plt.legend()
    plt.grid(True)
    plt.show()


# ============================================================
# Snapshot plots: 2x3, two scenarios (row 1 vs row 2)
# ============================================================

def _hex_to_rgb01(hex_color: str):
    hex_color = hex_color.lstrip("#")
    return (int(hex_color[0:2], 16) / 255.0,
            int(hex_color[2:4], 16) / 255.0,
            int(hex_color[4:6], 16) / 255.0)


def _blend_with_white(hex_color: str, intensity: float):
    """Blend base color with white.
    intensity=1 -> base color (darker/saturated),
    intensity=0 -> white (lighter).
    """
    intensity = float(min(1.0, max(0.0, intensity)))
    r, g, b = _hex_to_rgb01(hex_color)
    return (1.0 * (1.0 - intensity) + r * intensity,
            1.0 * (1.0 - intensity) + g * intensity,
            1.0 * (1.0 - intensity) + b * intensity)


def node_integration_ratio(G, i):
    """Local structural integration: fraction of neighbors from the other group."""
    neigh = list(G.neighbors(i))
    if not neigh:
        return 0.0
    t_i = G.nodes[i]["type"]
    cross = sum(1 for j in neigh if G.nodes[j]["type"] != t_i)
    return cross / len(neigh)


def plot_snapshot_2x3_structural(snaps_top, snaps_bottom, ts,
                                 row_titles=("Scenario 1: Rewiring ON", "Scenario 2: Rewiring OFF"),
                                 filename=None, dpi=300,
                                 min_intensity=0.25,
                                 title="Structural snapshots",
                                 fontsize_t=8):
    """Structural view (2x3):
    - node hue encodes group (HOST_COLOR / GUEST_COLOR)
    - node intensity encodes integration (darker = more integrated, lighter = more segregated)
    """
    from matplotlib.patches import Patch

    fig, axs = plt.subplots(2, 3, figsize=(12, 7), dpi=dpi)

    def draw_cell(ax, Gt, pos, t):
        # Integration ratio in [0,1]
        integ = {i: node_integration_ratio(Gt, i) for i in Gt.nodes()}
        # Intensity in [min_intensity, 1]: darker=more integrated
        intens = {i: (min_intensity + (1.0 - min_intensity) * integ[i]) for i in Gt.nodes()}

        node_colors = []
        for i in Gt.nodes():
            base = HOST_COLOR if Gt.nodes[i]["type"] == "host" else GUEST_COLOR
            node_colors.append(_blend_with_white(base, intens[i]))

        nx.draw_networkx_edges(Gt, pos, ax=ax, edge_color="0.7", alpha=0.25, width=0.5)
        nx.draw_networkx_nodes(Gt, pos, ax=ax, node_color=node_colors, node_size=22, linewidths=0)

        ax.set_title(fr"$t={t}$", fontsize=fontsize_t)
        ax.set_axis_off()

    # Top row
    for c, t in enumerate(ts):
        Gt, pos = snaps_top[t]
        draw_cell(axs[0, c], Gt, pos, t)

    # Bottom row
    for c, t in enumerate(ts):
        Gt, pos = snaps_bottom[t]
        draw_cell(axs[1, c], Gt, pos, t)

    # Row titles (on the left of first column, vertical)
    axs[0, 0].text(-0.07, 0.5, row_titles[0], transform=axs[0, 0].transAxes,
                   rotation=90, va="center", ha="right", fontsize=9)
    axs[1, 0].text(-0.07, 0.5, row_titles[1], transform=axs[1, 0].transAxes,
                   rotation=90, va="center", ha="right", fontsize=9)

    # Legend: show light vs dark for each group
    host_light = _blend_with_white(HOST_COLOR, min_intensity)
    host_dark = _blend_with_white(HOST_COLOR, 1.0)
    guest_light = _blend_with_white(GUEST_COLOR, min_intensity)
    guest_dark = _blend_with_white(GUEST_COLOR, 1.0)

    legend_elements = [
        Patch(facecolor=host_light, edgecolor='none', label="Host (more segregated)"),
        Patch(facecolor=host_dark, edgecolor='none', label="Host (more integrated)"),
        Patch(facecolor=guest_light, edgecolor='none', label="Guest (more segregated)"),
        Patch(facecolor=guest_dark, edgecolor='none', label="Guest (more integrated)"),
    ]
    fig.suptitle(title, fontsize=10, y=0.992)
    # Legend placed below the title (inside the canvas)
    fig.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, 0.965),
               ncol=4, frameon=False, fontsize=9)

    # Reserve top space for title + legend (avoid overlap/clipping)
    plt.tight_layout(rect=[0.03, 0.00, 1.00, 0.90])

    if filename:
        fig.savefig(filename, bbox_inches="tight")
        print(f"Saved figure to: {filename}")
    return fig


def plot_snapshot_2x3_cognitive(snaps_top, snaps_bottom, ts, lam=LAMBDA_SCORE,
                                vmin=-1, vmax=1,
                                filename=None, dpi=300,
                                row_titles=("Scenario 1: Rewiring ON", "Scenario 2: Rewiring OFF"),
                                title="Cognitive snapshots",
                                fontsize_t=8):
    """Cognitive view (2x3):
    - node color encodes neutrosophic scalar score Score_lambda (continuous)
    - colorbar annotated with semantic labels (negative/neutral/positive)
    """
    fig, axs = plt.subplots(2, 3, figsize=(12, 7), dpi=dpi)

    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.coolwarm

    def draw_cell(ax, Gt, pos, t):
        scores = [score_lambda(Gt.nodes[i]["attitude"], lam) for i in Gt.nodes()]
        colors = [cmap(norm(s)) for s in scores]
        nx.draw_networkx_edges(Gt, pos, ax=ax, edge_color="0.7", alpha=0.25, width=0.5)
        nx.draw_networkx_nodes(Gt, pos, node_color=colors, node_size=22, linewidths=0, ax=ax)
        ax.set_title(fr"$t={t}$", fontsize=fontsize_t)
        ax.set_axis_off()

    for c, t in enumerate(ts):
        Gt, pos = snaps_top[t]
        draw_cell(axs[0, c], Gt, pos, t)

    for c, t in enumerate(ts):
        Gt, pos = snaps_bottom[t]
        draw_cell(axs[1, c], Gt, pos, t)

    axs[0, 0].text(-0.07, 0.5, row_titles[0], transform=axs[0, 0].transAxes,
                   rotation=90, va="center", ha="right", fontsize=9)
    axs[1, 0].text(-0.07, 0.5, row_titles[1], transform=axs[1, 0].transAxes,
                   rotation=90, va="center", ha="right", fontsize=9)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    # Put the colorbar outside the subplot grid (no overlap)
    cax = fig.add_axes([0.92, 0.18, 0.02, 0.64])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(r"Neutrosophic attitude score (Score$_\lambda$)", fontsize=10)
    cbar.set_ticks([vmin, 0.0, vmax])
    cbar.set_ticklabels(["Negative", "Neutral", "Positive"])

    fig.suptitle(title, fontsize=10, y=0.96)
    # Reserve space on the right for the colorbar
    plt.tight_layout(rect=[0.03, 0.00, 0.90, 0.95])

    if filename:
        fig.savefig(filename, bbox_inches="tight")
        print(f"Saved figure to: {filename}")
    return fig


# ============================================================
# Snapshot plots: 2x3 for ONE scenario, showing BOTH views
#   Row 1: Structural (group colors; intensity = integration)
#   Row 2: Cognitive   (node color = Score_lambda)
# ============================================================

def plot_snapshot_2x3_scenario_struct_and_cog(snaps, ts,
                                              scenario_title="Scenario",
                                              filename=None, dpi=300,
                                              min_intensity=0.25,
                                              lam=LAMBDA_SCORE,
                                              vmin=-1, vmax=1,
                                              fontsize_t=8):
    """
    Paper-ready figure for a SINGLE scenario:
      - 2 rows x 3 columns
      - Columns: time snapshots (ts)
      - Row 1: structural view (HOST_COLOR/GUEST_COLOR; intensity=integration)
      - Row 2: cognitive view (Score_lambda colormap + colorbar)

    Legend (structural) is placed inside the canvas (top band);
    colorbar (cognitive) is placed outside on the right (no overlap).
    """
    from matplotlib.patches import Patch

    fig, axs = plt.subplots(2, 3, figsize=(12, 7), dpi=dpi)

    # ---------- structural row ----------
    def draw_struct(ax, Gt, pos, t):
        integ = {i: node_integration_ratio(Gt, i) for i in Gt.nodes()}
        intens = {i: (min_intensity + (1.0 - min_intensity) * integ[i]) for i in Gt.nodes()}

        node_colors = []
        for i in Gt.nodes():
            base = HOST_COLOR if Gt.nodes[i]["type"] == "host" else GUEST_COLOR
            node_colors.append(_blend_with_white(base, intens[i]))

        nx.draw_networkx_edges(Gt, pos, ax=ax, edge_color="0.7", alpha=0.25, width=0.5)
        nx.draw_networkx_nodes(Gt, pos, ax=ax, node_color=node_colors, node_size=22, linewidths=0)
        ax.set_title(fr"$t={t}$", fontsize=fontsize_t)
        ax.set_axis_off()

    # ---------- cognitive row ----------
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.coolwarm

    def draw_cog(ax, Gt, pos, t):
        scores = [score_lambda(Gt.nodes[i]["attitude"], lam) for i in Gt.nodes()]
        colors = [cmap(norm(s)) for s in scores]
        nx.draw_networkx_edges(Gt, pos, ax=ax, edge_color="0.7", alpha=0.25, width=0.5)
        nx.draw_networkx_nodes(Gt, pos, node_color=colors, node_size=22, linewidths=0, ax=ax)
        ax.set_title(fr"$t={t}$", fontsize=fontsize_t)
        ax.set_axis_off()

    # draw columns
    for c, t in enumerate(ts):
        Gt, pos_t = snaps[t]
        draw_struct(axs[0, c], Gt, pos_t, t)
        draw_cog(axs[1, c], Gt, pos_t, t)

    # Row labels (left side)
    axs[0, 0].text(-0.07, 0.5, "Structural", transform=axs[0, 0].transAxes,
                   rotation=90, va="center", ha="right", fontsize=9)
    axs[1, 0].text(-0.07, 0.5, "Cognitive", transform=axs[1, 0].transAxes,
                   rotation=90, va="center", ha="right", fontsize=9)

    # Structural legend: light vs dark for each group
    host_light = _blend_with_white(HOST_COLOR, min_intensity)
    host_dark = _blend_with_white(HOST_COLOR, 1.0)
    guest_light = _blend_with_white(GUEST_COLOR, min_intensity)
    guest_dark = _blend_with_white(GUEST_COLOR, 1.0)

    legend_elements = [
        Patch(facecolor=host_light, edgecolor='none', label="Host (more segregated)"),
        Patch(facecolor=host_dark, edgecolor='none', label="Host (more integrated)"),
        Patch(facecolor=guest_light, edgecolor='none', label="Guest (more segregated)"),
        Patch(facecolor=guest_dark, edgecolor='none', label="Guest (more integrated)"),
    ]

    # Titles: scenario as suptitle; legend just below inside the canvas
    fig.suptitle(scenario_title, fontsize=11, y=0.985)
    fig.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, 0.955),
               ncol=4, frameon=False, fontsize=9)

    # Cognitive colorbar (outside right)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cax = fig.add_axes([0.92, 0.14, 0.02, 0.30])  # only for cognitive row band
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(r"Neutrosophic attitude score (Score$_\lambda$)", fontsize=10)
    cbar.set_ticks([vmin, 0.0, vmax])
    cbar.set_ticklabels(["Negative", "Neutral", "Positive"])

    # Layout: reserve top for title+legend, right for colorbar
    plt.tight_layout(rect=[0.03, 0.00, 0.90, 0.90])

    if filename:
        fig.savefig(filename, bbox_inches="tight")
        print(f"Saved figure to: {filename}")
    return fig


# ============================================================
# Main run: generate snapshots and plot (always saves PNGs)
# ============================================================


# ============================================================
# Time-series plots (scores + integration metrics), saved to files
# ============================================================

def save_score_evolution_plot(scores_over_time, scenario_title, filename, dpi=300):
    """
    Plot evolution of scalar scores over time as distributions (histograms at selected steps).
    Saves to filename.
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)
    bins = np.linspace(-1, 1, 30)

    T = len(scores_over_time)
    # Use a few representative timesteps (start, ~10%, ~50%, end)
    steps = sorted(set([0, max(1, T // 10), max(1, T // 2), T - 1]))

    for t in steps:
        ax.hist(scores_over_time[t], bins=bins, alpha=0.5, label=f"t={t}")

    ax.set_title(scenario_title + " — score distribution snapshots", fontsize=11)
    ax.set_xlabel(r"Score$_\lambda$")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(True)

    fig.savefig(filename, bbox_inches="tight")
    print(f"Saved figure to: {filename}")
    return fig


def save_integration_metrics_plot(I_int_series, v_out_series, scenario_title, filename, dpi=300):
    """
    Plot integration metrics over time and save to filename.
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)

    ax.plot(I_int_series, label="I_int (cross-edge fraction)")
    ax.plot(v_out_series, label="v_out (cross reward share)")

    ax.set_title(scenario_title + " — integration metrics over time", fontsize=11)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True)

    fig.savefig(filename, bbox_inches="tight")
    print(f"Saved figure to: {filename}")
    return fig


if __name__ == "__main__":
    # Use sparse snapshots to keep runtime reasonable.
    TIMESTEPS_VIS = TIMESTEPS  # keep snapshots consistent with the simulation horizon
    SNAPSHOT_TS = [0, max(1, TIMESTEPS_VIS // 10), TIMESTEPS_VIS]
    print("Snapshot timesteps:", SNAPSHOT_TS)

    # Scenario 1: rewiring ON
    snaps_on = simulate_and_store_snapshots(
        timesteps=TIMESTEPS_VIS,
        snapshot_ts=SNAPSHOT_TS,
        seed=SEED,
        do_rewire=True,
        layout_iterations=10,
    )

    # Scenario 2: rewiring OFF (fixed edges)
    snaps_off = simulate_and_store_snapshots(
        timesteps=TIMESTEPS_VIS,
        snapshot_ts=SNAPSHOT_TS,
        seed=SEED + 1,
        do_rewire=False,
        layout_iterations=10,
    )

    print("Working directory:", __import__("os").getcwd())

    # Two figures: one per scenario, each contains Structural (row 1) + Cognitive (row 2)
    out_on = "snapshot_scenario1_rewiring_on.png"
    out_off = "snapshot_scenario2_rewiring_off.png"

    plot_snapshot_2x3_scenario_struct_and_cog(
        snaps_on, SNAPSHOT_TS,
        scenario_title="Scenario 1: Rewiring ON",
        filename=out_on
    )

    plot_snapshot_2x3_scenario_struct_and_cog(
        snaps_off, SNAPSHOT_TS,
        scenario_title="Scenario 2: Rewiring OFF",
        filename=out_off
    )

    # --------------------------------------------------------
    # Time-series outputs (scores + integration metrics) for both scenarios
    # --------------------------------------------------------
    # Scenario 1 time series (rewiring ON)
    _, scores_on, I_on, v_on = simulate_model_scenario(
        num_nodes=NUM_NODES,
        num_guests=NUM_GUESTS,
        timesteps=TIMESTEPS,
        initial_connection_prob=INITIAL_CONNECTION_PROB,
        lam=LAMBDA_SCORE,
        seed=SEED,
        do_rewire=True
    )

    # Scenario 2 time series (rewiring OFF)
    _, scores_off, I_off, v_off = simulate_model_scenario(
        num_nodes=NUM_NODES,
        num_guests=NUM_GUESTS,
        timesteps=TIMESTEPS,
        initial_connection_prob=INITIAL_CONNECTION_PROB,
        lam=LAMBDA_SCORE,
        seed=SEED + 1,
        do_rewire=False
    )

    out_scores_on = "scores_evolution_scenario1_rewiring_on.png"
    out_scores_off = "scores_evolution_scenario2_rewiring_off.png"
    out_metrics_on = "metrics_scenario1_rewiring_on.png"
    out_metrics_off = "metrics_scenario2_rewiring_off.png"

    save_score_evolution_plot(scores_on, "Scenario 1: Rewiring ON", out_scores_on)
    save_score_evolution_plot(scores_off, "Scenario 2: Rewiring OFF", out_scores_off)

    save_integration_metrics_plot(I_on, v_on, "Scenario 1: Rewiring ON", out_metrics_on)
    save_integration_metrics_plot(I_off, v_off, "Scenario 2: Rewiring OFF", out_metrics_off)

    print("Execution finished. If you didn't see windows, check the saved PNG files.")
    plt.show()