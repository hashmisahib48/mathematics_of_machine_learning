# Problem figures of Chapter 38. Run: python3 fig/probs.py  (answers -> verify/problem_answers.json)
from common import *
import scipy.cluster.hierarchy as sch
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.cluster import DBSCAN, HDBSCAN, cluster_optics_dbscan
from sklearn.datasets import load_wine
from birds import birds
plt.rcParams.update({'xtick.labelsize': 11, 'ytick.labelsize': 11, 'axes.labelsize': 11, 'legend.fontsize': 11})
A = {}; CHK = {}
def small(w=3.0, h=1.4): return plt.subplots(figsize=(w, h), dpi=240)
def sv(fig, name):
    fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
    fig.tight_layout(pad=0.3); save(fig, name)
    from PIL import Image
    with Image.open(name) as im: w, h = im.size
    CHK[name] = round(fs * min(1.55 * 135, 235 * w / h) / 96 / (w / 240), 1)
X, y = birds()

# 38A.1 core, border, noise on a line
x1 = np.array([0, 1, 2, 3.5, 5, 6, 9.]); eps1, m1 = 1.5, 3
D1 = np.abs(x1[:, None] - x1[None]); cnt = (D1 <= eps1).sum(1); core1 = cnt >= m1
lab1 = DBSCAN(eps=eps1, min_samples=m1).fit(x1[:, None]).labels_
A['38A.1'] = {'counts': cnt.tolist(), 'core': np.flatnonzero(core1).tolist(), 'labels': lab1.tolist(),
              'border': [i for i in range(7) if lab1[i] >= 0 and not core1[i]]}
fig, ax = small(3.0, 1.1)
for i, v in enumerate(x1): ax.scatter(v, 0, s=40, color=BLUE, zorder=5, edgecolor='white', lw=0.7); ax.text(v, 0.35, str(i + 1), ha='center', fontsize=11, color=NAVY)
ax.set_ylim(-0.5, 0.9); ax.set_yticks([]); ax.set_xticks(range(0, 10, 3)); ax.set_xlabel('position (km)'); ax.spines['left'].set_visible(False); sv(fig, 'p_38A1.png')

# 38A.2 mutual reachability on a line
x2 = np.array([0, 1, 1.5, 4.]); D2 = np.abs(x2[:, None] - x2[None]); c2 = np.sort(D2, 1)[:, 2]
MR = np.maximum(D2, np.maximum(c2[:, None], c2[None])); T = minimum_spanning_tree(MR).toarray()
A['38A.2'] = {'core_dist': c2.tolist(), 'MR': MR.tolist(), 'mst_weights': sorted(T[T > 0].tolist()), 'mst_total': float(T.sum()),
              'raw_mst_total': float(minimum_spanning_tree(D2).toarray().sum())}
fig, ax = small(3.0, 1.1)
for i, v in enumerate(x2): ax.scatter(v, 0, s=40, color=GREEN, zorder=5, edgecolor='white', lw=0.7); ax.text(v, 0.35, 'abcd'[i], ha='center', fontsize=11, color=NAVY)
ax.set_ylim(-0.5, 0.9); ax.set_yticks([]); ax.set_xticks(range(0, 5)); ax.set_xlabel('position'); ax.spines['left'].set_visible(False); sv(fig, 'p_38A2.png')

# 38A.3 reading a reachability plot
reach = np.array([np.inf, .3, .4, 1.2, .3, .35, .9, .4]); core4 = np.array([.3, .3, .4, .5, .3, .3, .45, .4]); order = np.arange(8)
res = {}
for e in (0.5, 1.0, 1.3):
    l = cluster_optics_dbscan(reachability=reach, core_distances=core4, ordering=order, eps=e); res[str(e)] = l.tolist()
A['38A.3'] = res
fig, ax = small(3.0, 1.4)
ax.bar(range(1, 9), np.where(np.isinf(reach), 1.5, reach), color=PUR, width=0.7); ax.axhline(0.5, color=RED, lw=1.0, ls='--'); ax.set_xlabel('position in the ordering'); ax.set_ylabel('reachability'); ax.set_ylim(0, 1.5); grid(ax); sv(fig, 'p_38A3.png')

# 38B.1 DBSCAN with m = 1 is single linkage
Z = sch.linkage(X, 'single'); rows = {}
for e in (0.1, 0.2, 0.3):
    a = DBSCAN(eps=e, min_samples=1).fit(X).labels_; b = sch.fcluster(Z, e, 'distance') - 1
    from sklearn.metrics import adjusted_rand_score as ari
    rows[str(e)] = {'dbscan_clusters': int(a.max() + 1), 'single_clusters': int(b.max() + 1), 'ari': round(float(ari(a, b)), 3)}
A['38B.1'] = rows
fig, ax = small(); es = np.arange(0.04, 0.42, 0.02); ax.plot(es, [DBSCAN(eps=e, min_samples=1).fit(X).labels_.max() + 1 for e in es], '-o', ms=3, color=BLUE, lw=1.3)
ax.set_yscale('log'); ax.set_xlabel('$\\varepsilon$'); ax.set_ylabel('clusters'); grid(ax); sv(fig, 'p_38B1.png')

# 38B.2 scale of the wine columns
Xw, yw = load_wine(return_X_y=True); sd = Xw.std(0); share = sd ** 2 / (sd ** 2).sum()
A['38B.2'] = {'proline_share_of_variance': round(float(share.max()), 4), 'second_share': round(float(np.sort(share)[-2]), 5),
              'ratio_sd': round(float(sd.max() / sd.min()), 0), 'scaled_share': round(1 / 13, 4)}
fig, ax = small(); ax.bar(range(13), sd, color=GREY, width=0.7); ax.bar([int(np.argmax(sd))], [sd.max()], color=OR, width=0.7)
ax.set_yscale('log'); ax.set_xlabel('column'); ax.set_ylabel('std'); ax.set_xticks([0, 4, 8, 12]); grid(ax); sv(fig, 'p_38B2.png')

# 38B.3 border points versus min_samples
ms = [3, 5, 8, 12]; rows = []
for m in ms:
    d = DBSCAN(eps=0.2, min_samples=m).fit(X); cs = np.zeros(len(X), bool); cs[d.core_sample_indices_] = True
    rows.append({'m': m, 'core': int(cs.sum()), 'border': int(((d.labels_ >= 0) & ~cs).sum()), 'noise': int((d.labels_ == -1).sum()), 'clusters': int(d.labels_.max() + 1)})
A['38B.3'] = rows
fig, ax = small(); ax.plot(ms, [r['border'] for r in rows], '-s', color=OR, ms=4, lw=1.3, label='border'); ax.plot(ms, [r['noise'] for r in rows], '-o', color=NOISE, ms=4, lw=1.3, label='noise')
ax.set_xlabel('min_samples'); ax.set_xticks(ms); ax.legend(loc='upper left', fontsize=11); grid(ax); sv(fig, 'p_38B3.png')

# 38C.1 a border point between two clusters
xc = np.r_[np.arange(0, .5, .1), 0.9, np.arange(1.4, 1.9, .1)]
dc = DBSCAN(eps=0.5, min_samples=5).fit(xc[:, None]); A['38C.1'] = {'labels': dc.labels_.tolist(), 'core': dc.core_sample_indices_.tolist()}
fig, ax = small(3.0, 1.1); ax.scatter(xc, 0 * xc, s=34, color=BLUE, zorder=5, edgecolor='white', lw=0.7); ax.scatter([0.9], [0], s=60, facecolor='white', edgecolor=OR, lw=1.6, zorder=6)
ax.set_ylim(-0.5, 0.8); ax.set_yticks([]); ax.set_xlabel('position'); ax.spines['left'].set_visible(False); sv(fig, 'p_38C1.png')

# 38C.2 one blob
one = np.random.default_rng(0).normal(size=(150, 2)); a = HDBSCAN(min_cluster_size=10, copy=False).fit(one)
A['38C.2'] = {'clusters': int(a.labels_.max() + 1), 'noise': int((a.labels_ == -1).sum())}
fig, ax = small(3.0, 1.4); scatter_labels(ax, one, a.labels_, s=9, noise_s=9); ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$'); grid(ax); sv(fig, 'p_38C2.png')

json.dump(A, open('verify/problem_answers.json', 'w'), indent=1); json.dump(CHK, open('verify/probfig_textsize.json', 'w'), indent=1)
print({k: v for k, v in CHK.items() if v < 7.5}); print(json.dumps(A))
