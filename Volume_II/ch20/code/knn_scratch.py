# k-nearest-neighbour classifier from scratch (brute force).
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


class MyKNN:
    """Brute-force k-NN; weights is 'uniform' or 'distance'."""

    def __init__(self, n_neighbors=5, weights="uniform"):
        self.n_neighbors = n_neighbors
        self.weights = weights

    def fit(self, X, y):
        self.X_ = np.asarray(X, dtype=float)
        self.classes_, self.y_ = np.unique(y, return_inverse=True)
        return self

    def kneighbors(self, X):
        diff = np.asarray(X, float)[:, None] - self.X_[None]
        d2 = (diff ** 2).sum(axis=2)       # Eq. 20.3-1 with p = 2
        k = self.n_neighbors
        idx = np.argsort(d2, axis=1, kind="stable")[:, :k]
        return np.sqrt(np.take_along_axis(d2, idx, axis=1)), idx

    def predict_proba(self, X):
        dist, idx = self.kneighbors(X)
        if self.weights == "uniform":
            w = np.ones_like(dist)         # Eq. 20.3-5
        else:
            with np.errstate(divide="ignore"):
                w = 1.0 / dist             # Eq. 20.3-5
            hit = np.isinf(w).any(axis=1)  # query on a data point:
            w[hit] = np.isinf(w[hit])      # only d = 0 points count
        votes = np.stack([(w * (self.y_[idx] == c)).sum(axis=1)
                          for c in range(len(self.classes_))], 1)
        total = votes.sum(axis=1, keepdims=True)    # Eq. 20.3-4
        return votes / total                        # Eq. 20.3-6

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
