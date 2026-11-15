# Problem 38B.1: DBSCAN with min_samples=1 is single linkage.
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.metrics import adjusted_rand_score as ari

# the 252 birds: two crescents of 120 and 12 uniform strays
X, _ = make_moons(n_samples=240, noise=0.06, random_state=0)
rng = np.random.default_rng(1)
X = np.r_[X, rng.uniform([-1.4, -0.9], [2.4, 1.4], (12, 2))]

Z = linkage(X, "single")              # the single-linkage tree
for eps in (0.1, 0.2, 0.3):
    a = DBSCAN(eps=eps, min_samples=1).fit(X).labels_
    b = fcluster(Z, eps, "distance") - 1    # cut at height eps
    size = np.bincount(a)
    print(f"eps={eps}: DBSCAN {a.max()+1}, single {b.max()+1} "
          f"clusters, ARI {ari(a, b):.3f}, noise {(a < 0).sum()}")
    print(f"   clusters of one sample {(size == 1).sum()}, "
          f"largest cluster {size.max()}")
