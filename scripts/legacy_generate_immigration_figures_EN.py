import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from pathlib import Path

OUT = Path('/mnt/data/grafici')
OUT.mkdir(exist_ok=True)
N=80
NG=16
T=150
p0=0.07
sigma=0.5
kappa=0.035
lam=0.6
HOST='#2ca02c'
GUEST='#9467bd'

def clamp_triplet(a):
    return tuple(float(min(1,max(0,x))) for x in a)

def score(a):
    T0,I,F=a
    return T0-lam*F-(1-lam)*I

def utility(a,b):
    d=abs(score(a)-score(b))
    return float(np.exp(-(d*d)/(2*sigma*sigma)))

def init(seed=42):
    rng=np.random.default_rng(seed)
    G=nx.erdos_renyi_graph(N,p0,seed=seed)
    guests=set(rng.choice(list(G.nodes()), size=NG, replace=False))
    for i in G.nodes():
        if i in guests:
            G.nodes[i]['type']='guest'
            G.nodes[i]['attitude']=clamp_triplet((rng.uniform(.05,.35),rng.uniform(.55,.90),rng.uniform(.55,.90)))
        else:
            G.nodes[i]['type']='host'
            G.nodes[i]['attitude']=clamp_triplet((rng.uniform(.65,.95),rng.uniform(.05,.30),rng.uniform(.05,.30)))
    return G

def update(G):
    new={}
    for i in G.nodes():
        a=np.array(G.nodes[i]['attitude'])
        neigh=list(G.neighbors(i))
        if not neigh:
            new[i]=tuple(a); continue
        drift=np.zeros(3)
        for j in neigh:
            b=np.array(G.nodes[j]['attitude'])
            drift += utility(a,b)*(b-a)
        new[i]=clamp_triplet(a+kappa*drift)
    for i,a in new.items():
        G.nodes[i]['attitude']=a

def rewire(G,rng):
    nodes=list(G.nodes())
    # sample non-edges and existing edges, not all pairs, for speed and softer dynamics
    edges=list(G.edges())
    for u,v in edges:
        if rng.random() > utility(G.nodes[u]['attitude'],G.nodes[v]['attitude']):
            G.remove_edge(u,v)
    # candidate additions
    for _ in range(N):
        u,v=rng.choice(nodes,2,replace=False)
        if not G.has_edge(u,v) and rng.random() < 0.15*utility(G.nodes[u]['attitude'],G.nodes[v]['attitude']):
            G.add_edge(u,v)

def metrics(G):
    m=G.number_of_edges()
    if m==0: return 0,0
    cross=0; num=0; den=0
    for u,v in G.edges():
        w=utility(G.nodes[u]['attitude'],G.nodes[v]['attitude'])
        den += w
        if G.nodes[u]['type'] != G.nodes[v]['type']:
            cross += 1; num += w
    return cross/m, (num/den if den>0 else 0)

def run(seed, do_rewire):
    rng=np.random.default_rng(seed)
    G=init(seed)
    scores=[]; I=[]; V=[]
    snapshots={}
    ts=[0,50,T]
    for t in range(T+1):
        if t in ts: snapshots[t]=G.copy()
        scores.append([score(G.nodes[i]['attitude']) for i in G.nodes()])
        a,b=metrics(G); I.append(a); V.append(b)
        if t<T:
            update(G)
            if do_rewire: rewire(G,rng)
    return G, scores, I, V, snapshots

def group_pos(G, seed=0):
    rng=np.random.default_rng(seed)
    pos={}
    for i in G.nodes():
        x0=-1.5 if G.nodes[i]['type']=='host' else 1.5
        pos[i]=(x0+rng.normal(0,.45), rng.normal(0,.65))
    return nx.spring_layout(G,pos=pos,iterations=30,seed=seed)

def save_snapshot(G,title,filename,seed):
    pos=group_pos(G,seed)
    colors=[HOST if G.nodes[i]['type']=='host' else GUEST for i in G.nodes()]
    fig,ax=plt.subplots(figsize=(7,5),dpi=180)
    nx.draw_networkx_edges(G,pos,ax=ax,edge_color='0.70',alpha=.35,width=.5)
    nx.draw_networkx_nodes(G,pos,ax=ax,node_color=colors,node_size=28,linewidths=0)
    ax.set_title(title)
    ax.set_axis_off()
    fig.savefig(OUT/filename,bbox_inches='tight')
    plt.close(fig)

def save_score(scores,title,filename):
    fig,ax=plt.subplots(figsize=(7,4.5),dpi=180)
    bins=np.linspace(-1,1,32)
    for t in [0,25,75,T]:
        ax.hist(scores[t],bins=bins,alpha=.45,label=f't={t}')
    ax.set_xlabel(r'$\mathrm{Sc}_\lambda$')
    ax.set_ylabel('frequency')
    ax.set_title(title)
    ax.grid(True,alpha=.3)
    ax.legend()
    fig.savefig(OUT/filename,bbox_inches='tight')
    plt.close(fig)

def save_metrics(I,V,title,filename):
    fig,ax=plt.subplots(figsize=(7,4.5),dpi=180)
    ax.plot(I,label=r'$I_{\mathrm{int}}$')
    ax.plot(V,label=r'$v_{\mathrm{out}}$')
    ax.set_xlabel('time')
    ax.set_ylabel('value')
    ax.set_ylim(-.02,1.02)
    ax.set_title(title)
    ax.grid(True,alpha=.3)
    ax.legend()
    fig.savefig(OUT/filename,bbox_inches='tight')
    plt.close(fig)

G_on,s_on,I_on,V_on,sn_on=run(42,True)
G_off,s_off,I_off,V_off,sn_off=run(43,False)
save_snapshot(G_on,'Scenario 1: rewiring ON','snapshot_scenario1_rewiring_on.png',1)
save_snapshot(G_off,'Scenario 2: rewiring OFF','snapshot_scenario2_rewiring_off.png',2)
save_score(s_on,'Scenario 1: score distribution','scores_evolution_scenario1_rewiring_on.png')
save_score(s_off,'Scenario 2: score distribution','scores_evolution_scenario2_rewiring_off.png')
save_metrics(I_on,V_on,'Scenario 1: integration metrics','metrics_scenario1_rewiring_on.png')
save_metrics(I_off,V_off,'Scenario 2: integration metrics','metrics_scenario2_rewiring_off.png')
print('done')
