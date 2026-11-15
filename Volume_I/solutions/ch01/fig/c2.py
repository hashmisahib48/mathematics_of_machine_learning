# Problem 1C.2: two ways of judging a clustering without labels. Writes fig/s1c2.png
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
from figstyle import *
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score, adjusted_rand_score
X, _ = make_blobs(n_samples=75, centers=[(-3, -1), (0, 3), (3, -1)], cluster_std=0.85, random_state=4)  # data of the problem
ks = [2, 3, 4, 5, 6]
inertia, sil, stab = [], [], []
rng = np.random.default_rng(0)
for k in ks:
    km = KMeans(k, n_init=10, random_state=0).fit(X)
    inertia.append(km.inertia_); sil.append(silhouette_score(X, km.labels_))
    aris = []                                  # stability: fit on two random halves, label all points with each, compare
    for rep in range(20):
        idx = rng.permutation(len(X)); a, b = idx[:37], idx[37:]
        la = KMeans(k, n_init=5, random_state=rep).fit(X[a]).predict(X)
        lb = KMeans(k, n_init=5, random_state=rep).fit(X[b]).predict(X)
        aris.append(adjusted_rand_score(la, lb))
    stab.append(float(np.mean(aris)))
print(" K  inertia  silhouette  stability(ARI between halves)")
for k, a, s, t in zip(ks, inertia, sil, stab): print(f"{k:2d} {a:8.1f} {s:10.3f} {t:10.3f}")
print("inertia never rises:", all(np.diff(inertia) < 0))
print("sizes of the 3 groups:", np.bincount(KMeans(3, n_init=10, random_state=0).fit_predict(X)).tolist())
fig, axs = plt.subplots(1, 3, figsize=(5.4, 1.8), dpi=240)
for ax, v, t, c, m in zip(axs, (inertia, sil, stab), ('inertia', 'silhouette', 'stability (ARI)'), (RED, GREEN, PUR), ('o', 's', '^')):
    ax.plot(ks, v, '-' + m, color=c, ms=4, lw=1.4); ax.set_xlabel('number of groups', fontsize=8.5); ax.set_title(t, fontsize=8.5, color=NAVY)
    ax.tick_params(labelsize=7.5); ax.grid(color=GRID, lw=0.5)
fig.tight_layout(w_pad=0.8); save(fig, 'fig/s1c2.png')
