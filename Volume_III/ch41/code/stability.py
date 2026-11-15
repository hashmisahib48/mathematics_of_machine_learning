# Stability under resampling: do two halves of the data agree?
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_wine, make_blobs
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.preprocessing import StandardScaler

wine = StandardScaler().fit_transform(load_wine().data)
cent = [(-5, -3), (-3, 4), (4, -4), (5, 3)]
blobs, _ = make_blobs(400, centers=cent, cluster_std=1.5,
                      random_state=3)


def stability(X, K, pairs=10, frac=0.8, seed=0):
    rng = np.random.default_rng(seed)
    n, scores = len(X), []
    for _ in range(pairs):
        ia = np.sort(rng.choice(n, int(frac * n), replace=False))
        ib = np.sort(rng.choice(n, int(frac * n), replace=False))
        la = KMeans(K, n_init=3, random_state=0).fit_predict(X[ia])
        lb = KMeans(K, n_init=3, random_state=1).fit_predict(X[ib])
        both = np.intersect1d(ia, ib)       # in both subsets
        scores.append(ari(la[np.searchsorted(ia, both)],
                          lb[np.searchsorted(ib, both)]))
    return np.mean(scores)


print("K   wine  blobs")
for K in range(2, 9):
    print(f"{K}  {stability(wine, K):5.3f}"
          f"  {stability(blobs, K):5.3f}")
