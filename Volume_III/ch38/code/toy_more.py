# Examples 38.7-2 and 38.7-3: mutual reachability and mean shift.
import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import HDBSCAN, MeanShift

X = np.array([[1, 1], [2, 1], [2, 2], [3, 2], [1, 2],
              [6, 1], [7, 1], [7, 2], [8, 2], [4, 5]], float)

# ---- mutual reachability of points 3, 4, 5 and 10 (min_samples = 3)
P = X[[2, 3, 4, 9]]
m = 3
D = np.sqrt(((P[:, None] - P[None]) ** 2).sum(-1))
core = np.sort(D, axis=1)[:, m - 1]            # Eq. 38.3-3
MR = np.maximum(np.maximum(core[:, None], core[None]), D)  # 38.4-1
np.fill_diagonal(MR, 0)
print("distances\n", np.round(D, 3))
print("core distances", np.round(core, 3))
print("mutual reachability\n", np.round(MR, 3))
Z = linkage(squareform(MR), method="single")
print("merge heights", np.round(Z[:, 2], 3))
print("raw merge heights",
      np.round(linkage(squareform(D), "single")[:, 2], 3))
h = HDBSCAN(min_cluster_size=2, min_samples=m, copy=False).fit(P)
sl = h._single_linkage_tree_              # private, for checking
assert np.allclose(sorted(sl["value"]), sorted(Z[:, 2]))
print("lambda = 1/height", np.round(1 / Z[:, 2], 4))

# ---- mean shift from point 4, flat kernel, bandwidth 1.5
bw = 1.5
x = X[3].copy()
for step in range(1, 4):
    d = np.linalg.norm(X - x, axis=1)
    inside = np.where(d <= bw)[0] + 1
    x = X[d <= bw].mean(axis=0)                # Eq. 38.6-5
    print("step", step, "points", inside, "mean", np.round(x, 4))
ms = MeanShift(bandwidth=bw).fit(X)
print("centres\n", ms.cluster_centers_, "\nlabels", ms.labels_)
