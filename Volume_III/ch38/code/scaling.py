# Scale decides the neighbourhood: DBSCAN and HDBSCAN on the wine data.
import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score as ari

X, y = load_wine(return_X_y=True)               # 178 x 13, hidden labels
Xs = StandardScaler().fit_transform(X)
sd = X.std(0)
print("largest and smallest std of a raw column:",
      f"{sd.max():.1f} (proline), {sd.min():.2f}")
for name, Z, grid in (("raw", X, [5, 10, 20, 40]),
                      ("scaled", Xs, [1.5, 2.0, 2.5, 3.0])):
    d = NearestNeighbors(n_neighbors=5).fit(Z).kneighbors(Z)[0][:, -1]
    print(f"{name:6s} 5-th neighbour distance, median {np.median(d):.2f},"
          f" 90% {np.percentile(d, 90):.2f}")
    for eps in grid:
        lab = DBSCAN(eps=eps, min_samples=5).fit(Z).labels_
        print(f"  eps={eps:<4} clusters={lab.max() + 1}"
              f" noise={(lab == -1).sum():3d} ARI={ari(y, lab):.3f}")
for name, Z in (("raw", X), ("scaled", Xs)):
    lab = DBSCAN().fit(Z).labels_
    print(f"default eps=0.5 on {name}: clusters={lab.max() + 1},"
          f" noise={(lab == -1).sum()}")
h = HDBSCAN(min_cluster_size=10, copy=False).fit(Xs)
print(f"HDBSCAN scaled, min_cluster_size=10: clusters={h.labels_.max() + 1},"
      f" noise={(h.labels_ == -1).sum()}, ARI={ari(y, h.labels_):.3f}")
