# Problem 38C.1: a border point that belongs to two clusters.
import numpy as np
from sklearn.cluster import DBSCAN, HDBSCAN

x = np.r_[np.arange(0, .5, .1), 0.9, np.arange(1.4, 1.9, .1)]   # 11 samples
print("positions", np.round(x, 1).tolist())
D = np.abs(x[:, None] - x[None]); eps, m = 0.5, 5
nb = D <= eps + 1e-12                                  # tolerance for the exact ties
cnt = nb.sum(1)
print("neighbourhood sizes", cnt.tolist(), "-> core:", np.round(x[cnt >= m], 1).tolist())
print("neighbours of 0.9:", np.round(x[nb[5]], 1).tolist(), " count", int(cnt[5]))
print("distances 0.9 - 0.4 and 1.4 - 0.9 in floating point:", repr(float(x[5] - x[4])), repr(float(x[6] - x[5])))
print("without the tolerance the sizes of 0.4 and 1.4 are:", int((D[4] <= eps).sum()), int((D[6] <= eps).sum()))
for name, o in (("given order", np.arange(11)), ("reversed", np.arange(11)[::-1])):
    d = DBSCAN(eps=eps, min_samples=m).fit(x[o, None])
    lab = d.labels_; back = np.empty(11, int); back[o] = lab
    print(f"{name:11s}: labels by position {back.tolist()}   0.9 -> cluster of {'0.4' if back[5] == back[4] else '1.4'}")
h = HDBSCAN(min_cluster_size=5, min_samples=5, copy=False).fit(x[:, None])
cut = h.dbscan_clustering(cut_distance=0.5, min_cluster_size=5)
print("dbscan_clustering(0.5, 5) of a fitted HDBSCAN:", cut.tolist(), " (border points become noise)")
rng = np.random.default_rng(0); left = 0
for _ in range(1000):
    o = rng.permutation(11); l = DBSCAN(eps=eps, min_samples=m).fit(x[o, None]).labels_
    pos = np.empty(11, int); pos[o] = np.arange(11)
    left += l[pos[5]] == l[pos[4]]
print("1000 random orders: 0.9 joins the left group in", left, "and the right group in", 1000 - left)
