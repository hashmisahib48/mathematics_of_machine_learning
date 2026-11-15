# Problem 38B.3: core, border, noise against min_samples.
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.neighbors import NearestNeighbors

# the 252 birds: two crescents of 120 and 12 uniform strays
X, y = make_moons(n_samples=240, noise=0.06, random_state=0)
rng = np.random.default_rng(1)
X = np.r_[X, rng.uniform([-1.4, -0.9], [2.4, 1.4], (12, 2))]
y = np.r_[y, np.full(12, -1)]               # -1: stray

eps = 0.2
nb = NearestNeighbors(radius=eps).fit(X).radius_neighbors(X)[0]
size = np.array([len(s) for s in nb])   # |N_eps|, self included
print("neighbourhood sizes: min", size.min(),
      "median", int(np.median(size)), "max", size.max())
print(" m  core border noise clusters  sizes        noise strays")
for m in (3, 5, 8, 12):
    db = DBSCAN(eps=eps, min_samples=m).fit(X)
    core = np.zeros(len(X), bool)
    core[db.core_sample_indices_] = True
    lab = db.labels_
    n_b = ((lab >= 0) & ~core).sum()
    sizes = np.sort(np.bincount(lab[lab >= 0]))[::-1]
    s = ((lab == -1) & (y == -1)).sum()
    print(f"{m:2d} {core.sum():5d} {n_b:6d} {(lab < 0).sum():5d} "
          f"{lab.max()+1:7d}  {sizes}  {s} of 12")

# at m = 12: members of each cluster (moon 0, moon 1, stray)
lab = DBSCAN(eps=eps, min_samples=12).fit(X).labels_
for k in range(lab.max() + 1):
    print("cluster", k,
          [int((y[lab == k] == g).sum()) for g in (0, 1, -1)])
