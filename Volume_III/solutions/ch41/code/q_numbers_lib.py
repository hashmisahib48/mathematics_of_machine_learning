# the stability function of ch41/code/stability.py, shared by the solution scripts
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics as m

def stability(X, K, pairs=10, frac=0.8, seed=0):          # as in ch41/code/stability.py
    rng = np.random.default_rng(seed); n = len(X); out = []
    for _ in range(pairs):
        ia = np.sort(rng.choice(n, int(frac * n), replace=False)); ib = np.sort(rng.choice(n, int(frac * n), replace=False))
        la = KMeans(K, n_init=3, random_state=0).fit_predict(X[ia]); lb = KMeans(K, n_init=3, random_state=1).fit_predict(X[ib])
        both = np.intersect1d(ia, ib)
        out.append(m.adjusted_rand_score(la[np.searchsorted(ia, both)], lb[np.searchsorted(ib, both)]))
    return np.mean(out)
