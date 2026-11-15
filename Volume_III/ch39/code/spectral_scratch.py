# Normalised spectral clustering from scratch (Section 39.8).
import numpy as np
from scipy.linalg import eigh
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.datasets import make_blobs
from data import rings

class MySpectral:
    def __init__(self, n_clusters=2, gamma=1.0, random_state=0):
        self.n_clusters, self.gamma = n_clusters, gamma
        self.random_state = random_state

    def fit(self, X):
        sq = ((X[:, None] - X[None]) ** 2).sum(-1)
        W = np.exp(-self.gamma * sq)            # Eq. 39.3-1
        np.fill_diagonal(W, 0.0)                # no self-loops
        d = W.sum(axis=1)                       # degrees
        Ls = np.eye(len(X)) - W / np.sqrt(np.outer(d, d))   # 39.3-8
        lam, V = eigh(Ls, subset_by_index=[0, self.n_clusters - 1])
        U = V / np.sqrt(d)[:, None]             # f = D^(-1/2) g
        km = KMeans(self.n_clusters, n_init=10,
                    random_state=self.random_state).fit(U)
        self.eigenvalues_, self.labels_ = lam, km.labels_
        return self

def same_partition(a, b):
    """True if a and b differ only by the names of the clusters."""
    return len(set(zip(a, b))) == len(set(a)) == len(set(b))

tests = [(rings()[0], 2, 30.0),
         (make_blobs(300, centers=3, random_state=2)[0], 3, 0.05)]
for X, K, g in tests:
    mine = MySpectral(K, g).fit(X)
    ref = SpectralClustering(K, gamma=g, random_state=0).fit(X)
    assert same_partition(mine.labels_, ref.labels_)
    print("K =", K, "sizes", np.bincount(mine.labels_), "lambda",
          mine.eigenvalues_.round(4), "same partition")
