# Problem figures of Chapter 39. Run: python3 fig/probs.py  (answers -> verify/problem_answers.json)
from common import *
import itertools
from scipy.linalg import eigh
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import SpectralClustering, SpectralCoclustering, KMeans
from sklearn.datasets import make_blobs, make_biclusters
from sklearn.metrics import adjusted_rand_score as ari, consensus_score
from sklearn.metrics.pairwise import rbf_kernel
from data import rings, TOY, toy_adjacency
warnings.simplefilter('ignore', UserWarning)
warnings.simplefilter('ignore', RuntimeWarning)
plt.rcParams.update({'xtick.labelsize': 11, 'ytick.labelsize': 11, 'axes.labelsize': 11, 'legend.fontsize': 11})
A = {}; CHK = {}


def small(w=3.0, h=1.1): return plt.subplots(figsize=(w, h), dpi=240)


def sv(fig, name):
    fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
    fig.tight_layout(pad=0.3); save(fig, name)
    from PIL import Image
    with Image.open(name) as im: w, h = im.size
    CHK[name] = round(fs * min(1.55 * 175, 235 * w / h) / 96 / (w / 240), 1)


def node_graph(ax, pos, edges, lab=None, cols=None):
    for i, j in edges: ax.plot(*np.array([pos[i], pos[j]]).T, color=GREY, lw=1.1, zorder=1)
    for i, p in enumerate(pos):
        ax.scatter(*p, s=50, color=(cols[i] if cols else BLUE), edgecolor='white', lw=0.7, zorder=5)
        ax.text(p[0], p[1] + 0.42, str(i + 1) if lab is None else lab[i], ha='center', fontsize=11, color=NAVY)


def ncut_of(W, Aset):
    d = W.sum(1); B = [i for i in range(len(W)) if i not in Aset]
    cut = W[np.ix_(Aset, B)].sum(); vA, vB = d[list(Aset)].sum(), d[B].sum()
    return float(cut), float(vA), float(vB), float(cut * (1 / vA + 1 / vB)), float(cut * (1 / len(Aset) + 1 / len(B)))


# 39A.1 a path of four nodes
Wp = np.zeros((4, 4))
for i in range(3): Wp[i, i + 1] = Wp[i + 1, i] = 1
rows = {}
for Aset in ([0], [0, 1], [0, 1, 2]):
    c, vA, vB, nc, rc = ncut_of(Wp, Aset)
    rows[str([a + 1 for a in Aset])] = {'cut': c, 'volA': vA, 'volB': vB, 'ncut': round(nc, 4), 'ratiocut': round(rc, 4)}
A['39A.1'] = {'degrees': Wp.sum(1).tolist(), 'splits': rows}
fig, ax = small(3.0, 0.9)
node_graph(ax, [(0, 0), (1, 0), (2, 0), (3, 0)], [(0, 1), (1, 2), (2, 3)])
ax.set_xlim(-0.4, 3.4); ax.set_ylim(-0.4, 1.0); ax.axis('off'); sv(fig, 'p_39A1.png')

# 39A.2 the quadratic form of the six-point graph
W6 = toy_adjacency(); d6 = W6.sum(1); L6 = np.diag(d6) - W6
f = np.array([1, 1, 1, -1, -1, -1.])
lam6, F6 = eigh(L6, np.diag(d6))
A['39A.2'] = {'fLf': float(f @ L6 @ f), 'fDf': float(f @ np.diag(d6) @ f), 'fD1': float(f @ d6), 'rayleigh': round(float(f @ L6 @ f / (f @ np.diag(d6) @ f)), 4),
              'ncut': round(ncut_of(W6, [0, 1, 2])[3], 4), 'lambda2': round(float(lam6[1]), 4)}
fig, ax = small(3.0, 1.15)
ax.bar(range(1, 7), f, color=[BLUE] * 3 + [OR] * 3, width=0.6); ax.axhline(0, color=GREY, lw=0.8)
ax.set_xticks(range(1, 7)); ax.set_xlabel('node'); ax.set_ylabel('$f_i$'); ax.set_ylim(-1.5, 1.5); grid(ax); sv(fig, 'p_39A2.png')

# 39A.3 the six-point graph without its link
W6b = W6.copy(); W6b[2, 3] = W6b[3, 2] = 0
d = W6b.sum(1); Lb = np.diag(d) - W6b; ev = np.linalg.eigvalsh(Lb)
comp = connected_components(W6b)
A['39A.3'] = {'eigenvalues': ev.round(4).tolist(), 'components': int(comp[0]), 'labels': comp[1].tolist()}
fig, ax = small(3.0, 1.1)
node_graph(ax, [tuple(p / 1.0) for p in TOY], [(i, j) for i in range(6) for j in range(i + 1, 6) if W6b[i, j]])
ax.set_xlim(-0.8, 7.8); ax.set_ylim(-0.6, 3.0); ax.set_aspect('equal'); ax.axis('off'); sv(fig, 'p_39A3.png')

# 39A.4 net similarity of exemplar sets, three points on a line
x = np.array([0., 1., 3.]); S = -(x[:, None] - x[None]) ** 2


def net(Eset, p):
    return sum(S[i, min(Eset, key=lambda k: -S[i, k])] if i not in Eset else p for i in range(3))


best = {}
for p in (-6.0, -2.5, -0.5):
    vals = {str([e + 1 for e in E]): net(E, p) for r in (1, 2, 3) for E in itertools.combinations(range(3), r)}
    best[str(p)] = {'best': max(vals, key=vals.get), 'net': max(vals.values()), 'all': vals}
grid_p = np.linspace(-8, 0, 801)
cnt = []
for p in grid_p:
    v = {r: max(net(E, p) for E in itertools.combinations(range(3), r)) for r in (1, 2, 3)}
    cnt.append(max(v, key=v.get))
cnt = np.array(cnt)
A['39A.4'] = {'optimal_sets': best, 'switch_1_to_2': round(float(grid_p[np.argmax(cnt >= 2)]), 2), 'switch_2_to_3': round(float(grid_p[np.argmax(cnt >= 3)]), 2),
              'lines': {'1 exemplar': '-5 + p', '2 exemplars': '-1 + 2p', '3 exemplars': '3p'}}
fig, ax = small(3.0, 1.45)
pp = np.linspace(-7, 0, 100)
ax.plot(pp, -5 + pp, color=BLUE, lw=1.4, label='1'); ax.plot(pp, -1 + 2 * pp, color=OR, lw=1.4, ls='--', label='2'); ax.plot(pp, 3 * pp, color=GREEN, lw=1.4, ls=':', label='3')
ax.set_xlabel('preference $p$'); ax.set_ylabel('net similarity'); ax.set_ylim(-22, 0.5); grid(ax); ax.legend(loc='lower right', ncol=3, handlelength=1.4, fontsize=9, borderaxespad=0.1, columnspacing=0.8, title='exemplars:', title_fontsize=9) if False else ax.legend(loc='lower right', ncol=3, handlelength=1.4, fontsize=9, borderaxespad=0.1, columnspacing=0.8); ax.text(-6.9, -3, 'lines: number of exemplars', fontsize=9); sv(fig, 'p_39A4.png')

# 39B.1 scan gamma on the rings
X, y = rings(); res = {}
gs = [0.3, 1, 3, 10, 30, 100, 300, 1000]
for g in gs:
    res[str(g)] = round(float(ari(y, SpectralClustering(2, gamma=g, random_state=0).fit_predict(X))), 3)
A['39B.1'] = res
fig, ax = small(); ax.semilogx(gs, [res[str(g)] for g in gs], '-o', ms=3.5, color=PUR, lw=1.4); ax.set_xlabel('$\\gamma$'); ax.set_ylabel('ARI'); ax.set_ylim(-0.1, 1.1); grid(ax); sv(fig, 'p_39B1.png')

# 39B.2 components of the nearest-neighbour graph
res = {}; ks = [3, 4, 5, 10, 20, 30, 50]
for k in ks:
    sp = SpectralClustering(2, affinity='nearest_neighbors', n_neighbors=k, random_state=0).fit(X)
    res[str(k)] = {'components': int(connected_components(sp.affinity_matrix_)[0]), 'ari': round(float(ari(y, sp.labels_)), 3)}
A['39B.2'] = res
fig, ax = small(); ax.step(ks, [res[str(k)]['components'] for k in ks], where='mid', color=RED, lw=1.4); ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xticks([3, 5, 10, 20, 50]); ax.set_xticklabels(['3', '5', '10', '20', '50']); ax.set_xlabel('$k$'); ax.set_ylabel('components'); ax.set_yticks([1, 3, 10, 20]); ax.set_yticklabels(['1', '3', '10', '20']); ax.minorticks_off(); grid(ax); sv(fig, 'p_39B2.png')

# 39B.3 consensus score against the number of biclusters asked for
Bm, rows_, cols_ = make_biclusters((80, 80), 3, noise=5, shuffle=False, random_state=0)
pr = np.random.default_rng(1).permutation(80); pc = np.random.default_rng(2).permutation(80)
Bs = Bm[pr][:, pc]; res = {}
for K in (2, 3, 4, 5):
    co = SpectralCoclustering(K, random_state=0).fit(Bs)
    res[str(K)] = round(float(consensus_score(co.biclusters_, (rows_[:, pr], cols_[:, pc]))), 3)
A['39B.3'] = res
fig, ax = small(); ax.bar([2, 3, 4, 5], [res[str(k)] for k in (2, 3, 4, 5)], color=[GREY, GREEN, GREY, GREY], width=0.6); ax.set_xticks([2, 3, 4, 5]); ax.set_xlabel('$K$'); ax.set_ylabel('score'); ax.set_ylim(0, 1.15); grid(ax); sv(fig, 'p_39B3.png')

# 39C.1 the null space of the Laplacian of two components
w_, V_ = np.linalg.eigh(Lb)
fig, ax = small(3.0, 1.15)
for i in range(6): ax.scatter(V_[i, 0], V_[i, 1], s=70 - 8 * (i % 3), color=BLUE if i < 3 else OR, marker='o' if i < 3 else 's', edgecolor='white', lw=0.6, zorder=5)
ax.set_xlim(-0.15, 0.75); ax.set_ylim(-0.15, 0.75); ax.set_xlabel('$f^{(1)}_i$'); ax.set_ylabel('$f^{(2)}_i$'); grid(ax); sv(fig, 'p_39C1.png')
A['39C.1'] = {'rows_of_null_space': V_[:, :2].round(4).tolist()}

# 39C.2 the default preference of affinity propagation
Xb, yb = make_blobs(90, centers=3, cluster_std=0.9, random_state=4)
Sb = -((Xb[:, None] - Xb[None]) ** 2).sum(-1)
fig, ax = small(3.0, 1.15)
ax.hist(-Sb[np.triu_indices(90, 1)], bins=40, color=BLUE, alpha=0.8)
ax.axvline(-np.median(Sb), color=RED, lw=1.4); ax.set_xlabel('$-s(i,k)$'); ax.set_ylabel('pairs'); grid(ax); sv(fig, 'p_39C2.png')
A['39C.2'] = {'median_all': round(float(np.median(Sb)), 3), 'median_off_diagonal': round(float(np.median(Sb[~np.eye(90, dtype=bool)])), 3)}
import json
json.dump(A, open('verify/problem_answers.json', 'w'), indent=1)
json.dump(CHK, open('verify/probfig_textsize.json', 'w'), indent=1)
print(json.dumps(A, indent=1)[:3500])
