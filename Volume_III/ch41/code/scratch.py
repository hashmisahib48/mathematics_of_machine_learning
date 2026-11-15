# ARI and silhouette from the formulas of Chapter 41, with a test.
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_wine
from sklearn.metrics import adjusted_rand_score, silhouette_samples
from sklearn.preprocessing import StandardScaler


def my_ari(truth, pred):
    _, t = np.unique(truth, return_inverse=True)
    _, u = np.unique(pred, return_inverse=True)
    N = np.zeros((t.max() + 1, u.max() + 1), dtype=np.int64)
    np.add.at(N, (t, u), 1)                      # contingency table
    n = N.sum()
    c2 = lambda m: (m * (m - 1) // 2).sum()      # sum of C(m, 2)
    P11, Pt, Pc = c2(N), c2(N.sum(1)), c2(N.sum(0))
    if Pt == Pc == P11:                          # identical
        return 1.0
    E = Pt * Pc / (n * (n - 1) / 2)              # Eq. 41.4-3
    return (P11 - E) / ((Pt + Pc) / 2 - E)       # Eq. 41.4-4


def my_silhouette_samples(X, labels):
    D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
    _, lab = np.unique(labels, return_inverse=True)
    size = np.bincount(lab)
    S = np.stack([D[:, lab == k].sum(1)
                  for k in range(len(size))], 1)
    idx = np.arange(len(lab))
    a = S[idx, lab] / np.maximum(size[lab] - 1, 1)   # Eq. 41.3-1
    M = S / size                                 # means to clusters
    M[idx, lab] = np.inf                         # not the own one
    b = M.min(axis=1)                            # Eq. 41.3-1
    with np.errstate(invalid="ignore", divide="ignore"):
        s = (b - a) / np.maximum(a, b)           # Eq. 41.3-2
    s[size[lab] == 1] = 0.0                      # singleton cluster
    return s


if __name__ == "__main__":
    X, y = load_wine(return_X_y=True)
    Xs = StandardScaler().fit_transform(X)
    lab = KMeans(3, n_init=10, random_state=0).fit_predict(Xs)
    ari, s = my_ari(y, lab), my_silhouette_samples(Xs, lab)
    assert np.isclose(ari, adjusted_rand_score(y, lab))
    assert np.allclose(s, silhouette_samples(Xs, lab))
    print(f"wine, K = 3: ARI {ari:.3f}, silhouette {s.mean():.3f}")
    rng = np.random.default_rng(0)
    for t in range(200):                         # random labellings
        u, v = rng.integers(0, 4, 40), rng.integers(0, 6, 40)
        assert np.isclose(my_ari(u, v), adjusted_rand_score(u, v))
        assert np.isclose(my_ari(u, 7 - v), my_ari(u, v))  # renamed
    print("200 random labellings agree, also when renamed")
