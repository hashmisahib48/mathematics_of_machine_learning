# Problem figures and answers of Chapter 36. Run: python3 fig/probs.py  (answers -> verify/problem_answers.json)
from common import *
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, make_circles, load_wine
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
plt.rcParams.update({'xtick.labelsize': 9, 'ytick.labelsize': 9, 'axes.labelsize': 9.5, 'legend.fontsize': 9})
A = {}; CHK = {}
def small(w=2.6, h=0.95): return plt.subplots(figsize=(w, h), dpi=240)
def sv2(fig, name):
    fs = min([t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible()] or [9]); save(fig, name)
    with Image.open(name) as im: w, h = im.size
    pw = min(1.55 * 135, 235 * w / h); CHK[name] = round(fs * pw / 96 / (w / 240), 1)

# 36A.1 one assignment, one update (1-D)
x = np.array([0, 2, 6, 12.]); mu = np.array([1., 10.])
d2 = (x[:, None] - mu[None]) ** 2; c = d2.argmin(1); J0 = float(d2[np.arange(4), c].sum())
mu1 = np.array([x[c == k].mean() for k in (0, 1)]); J1 = float(((x - mu1[c]) ** 2).sum())
d2b = (x[:, None] - mu1[None]) ** 2; c2 = d2b.argmin(1)
A['36A.1'] = {'labels': (c + 1).tolist(), 'J_start': J0, 'new_centres': mu1.tolist(), 'J_after_update': J1, 'labels_next': (c2 + 1).tolist()}
fig, ax = small(2.4, 0.85)
ax.scatter(x, [0] * 4, s=34, color=BLUE, zorder=5, edgecolor='white'); ax.scatter(mu, [0] * 2, s=60, marker='X', color=GREEN, zorder=6, edgecolor='white')
ax.axhline(0, color=GREY, lw=0.8); ax.set_yticks([]); ax.set_xticks([0, 4, 8, 12]); ax.set_ylim(-0.5, 0.5); ax.spines['left'].set_visible(False); ax.set_xlabel('$x$'); sv2(fig, 'p_36A1.png')

# 36A.2 SST = SSW + SSB
x2 = np.array([1, 3, 8, 10.]); sst = float(((x2 - x2.mean()) ** 2).sum()); ssw = float(((x2[:2] - 2) ** 2).sum() + ((x2[2:] - 9) ** 2).sum()); ssb = float(2 * (2 - x2.mean()) ** 2 + 2 * (9 - x2.mean()) ** 2)
A['36A.2'] = {'mean': float(x2.mean()), 'SST': sst, 'SSW': ssw, 'SSB': ssb}
fig, ax = small(2.4, 0.85)
ax.scatter(x2, [0] * 4, s=34, color=BLUE, zorder=5, edgecolor='white'); ax.scatter([2, 9], [0, 0], s=60, marker='X', color=GREEN, zorder=6, edgecolor='white')
ax.axvline(x2.mean(), color=PUR, lw=1.0, ls='--'); ax.axhline(0, color=GREY, lw=0.8); ax.set_yticks([]); ax.set_xticks([0, 5, 10]); ax.set_ylim(-0.5, 0.5); ax.spines['left'].set_visible(False); ax.set_xlabel('$x$'); sv2(fig, 'p_36A2.png')

# 36A.3 k-means++ probabilities
x3 = np.array([0, 1, 2, 10.]); d3 = (x3 - 0) ** 2; p3 = d3 / d3.sum()
A['36A.3'] = {'D2': d3.tolist(), 'sum': float(d3.sum()), 'p': [round(float(v), 4) for v in p3], 'p_far': round(float(p3[3]), 4)}
fig, ax = small(2.4, 0.95)
ax.bar(range(1, 5), p3, color=BLUE, width=0.6); ax.set_xticks(range(1, 5)); ax.set_xticklabels(['0', '1', '2', '10']); ax.set_xlabel('point $x$'); ax.set_ylabel('probability'); ax.set_ylim(0, 1.1); grid(ax); sv2(fig, 'p_36A3.png')

# 36A.4 mini-batch update
N = 6; m0 = np.array([2., 3.]); B = np.array([[5., 3.], [3., 7.]]); new = (N * m0 + B.sum(0)) / (N + len(B))
A['36A.4'] = {'new_centre': new.tolist(), 'step': float(len(B) / (N + len(B)))}
fig, ax = small(2.4, 1.5)
ax.scatter(*B.T, s=34, color=BLUE, zorder=5, edgecolor='white'); ax.scatter(*m0, s=60, marker='X', color=GREEN, zorder=6, edgecolor='white'); ax.scatter(*new, s=60, marker='X', facecolor='white', edgecolor=GREEN, lw=1.3, zorder=6)
ax.set_xlim(1, 6); ax.set_ylim(2, 8); ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$'); grid(ax); sv2(fig, 'p_36A4.png')

# 36B.1 inertia of the scaled wine data for K = 1..6
X = StandardScaler().fit_transform(load_wine().data); Jw = {}
for K in range(1, 7): Jw[K] = round(float(KMeans(K, n_init=10, random_state=0).fit(X).inertia_), 2)
A['36B.1'] = {'J': Jw, 'n_times_p': X.shape[0] * X.shape[1]}
fig, ax = small(); ax.plot(list(Jw), list(Jw.values()), '-o', color=RED, ms=4, lw=1.3); ax.set_xlabel('$K$'); ax.set_ylabel('inertia'); ax.set_ylim(0, None); grid(ax); sv2(fig, 'p_36B1.png')

# 36B.2 restarts on the eight blobs
Xb, _ = make_blobs(800, centers=8, cluster_std=0.9, center_box=(-10, 10), random_state=4); R = 50
Jr = np.array([KMeans(8, init='random', n_init=1, random_state=s).fit(Xb).inertia_ for s in range(R)])
Jp = np.array([KMeans(8, init='k-means++', n_init=1, random_state=s).fit(Xb).inertia_ for s in range(R)])
best = min(Jr.min(), Jp.min())
A['36B.2'] = {'best': round(float(best), 2), 'random_within_2pct': round(float((Jr < 1.02 * best).mean()), 2), 'kpp_within_2pct': round(float((Jp < 1.02 * best).mean()), 2), 'runs': R}
fig, ax = small(); ax.bar([0, 1], [(Jr < 1.02 * best).mean(), (Jp < 1.02 * best).mean()], color=[OR, BLUE], width=0.55); ax.set_xticks([0, 1]); ax.set_xticklabels(['random', 'k-means++']); ax.set_ylabel('share good'); ax.set_ylim(0, 1); grid(ax); sv2(fig, 'p_36B2.png')

# 36B.3 min-max scaling
Xr = load_wine().data; y = load_wine().target; res = {}
for nm, S in (('none', None), ('standard', StandardScaler()), ('minmax', MinMaxScaler())):
    Z = Xr if S is None else S.fit_transform(Xr); km = KMeans(3, n_init=10, random_state=0).fit(Z); res[nm] = round(float(adjusted_rand_score(y, km.labels_)), 3)
A['36B.3'] = res
fig, ax = small(); ax.bar(range(3), list(res.values()), color=[GREY, BLUE, OR], width=0.6); ax.set_xticks(range(3)); ax.set_xticklabels(['raw', 'standard', 'min-max'], fontsize=9); ax.set_ylabel('ARI'); ax.set_ylim(0, 1); grid(ax); sv2(fig, 'p_36B3.png')

# 36C.1 rings
Xc, yc = make_circles(300, factor=0.4, noise=0.05, random_state=0); km = KMeans(2, n_init=10, random_state=0).fit(Xc)
A['36C.1'] = {'ARI': round(float(adjusted_rand_score(yc, km.labels_)), 3)}
fig, ax = small(2.4, 1.7); pts(ax, Xc, km.labels_, s=7, truth=yc); centres(ax, km.cluster_centers_, 50); ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect('equal'); sv2(fig, 'p_36C1.png')

# 36C.2 an elbow that is not there
Xu = np.random.default_rng(0).uniform(size=(400, 2)); Ju = [KMeans(K, n_init=5, random_state=0).fit(Xu).inertia_ for K in range(1, 9)]
A['36C.2'] = {'J': [round(float(v), 1) for v in Ju]}
fig, ax = small(); ax.plot(range(1, 9), Ju, '-o', color=RED, ms=4, lw=1.3); ax.set_xlabel('$K$'); ax.set_ylabel('inertia'); ax.set_ylim(0, None); grid(ax); sv2(fig, 'p_36C2.png')

json.dump(A, open('verify/problem_answers.json', 'w'), indent=1); json.dump(CHK, open('verify/probfig_textsize.json', 'w'), indent=1)
print({k: v for k, v in CHK.items() if v < 7.5}); print(json.dumps(A))
