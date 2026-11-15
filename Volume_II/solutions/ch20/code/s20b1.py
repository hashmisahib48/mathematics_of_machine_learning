# Problem 20B.1: k-NN regression from scratch, Eq. 20.3-7
import numpy as np
from sklearn.datasets import make_regression
from sklearn.neighbors import KNeighborsRegressor


class MyKNNRegressor:
    def __init__(self, n_neighbors=5, weights="uniform"):
        self.n_neighbors, self.weights = n_neighbors, weights

    def fit(self, X, y):
        self.X_, self.y_ = (np.asarray(v, float) for v in (X, y))
        return self

    def predict(self, X):
        diff = np.asarray(X, float)[:, None] - self.X_[None]
        d2 = (diff ** 2).sum(axis=2)             # Eq. 20.3-1, q = 2
        idx = np.argsort(d2, 1, kind="stable")[:, :self.n_neighbors]
        dist = np.sqrt(np.take_along_axis(d2, idx, axis=1))
        w = np.ones_like(dist)                     # Eq. 20.3-5
        if self.weights == "distance":
            with np.errstate(divide="ignore"):
                w = 1.0 / dist
            hit = np.isinf(w).any(axis=1)  # query on a sample:
            w[hit] = np.isinf(w[hit])            # only d = 0 counts
        num = (w * self.y_[idx]).sum(axis=1)     # Eq. 20.3-7
        return num / w.sum(axis=1)

rng = np.random.default_rng(0)
X, y = make_regression(200, 4, noise=5, random_state=0)
Xq = rng.normal(size=(50, 4))
Xq[:3] = X[:3]               # three queries on training rows
for wt in ("uniform", "distance"):
    mine = MyKNNRegressor(5, wt).fit(X, y).predict(Xq)
    ref = KNeighborsRegressor(5, weights=wt).fit(X, y).predict(Xq)
    print(f"{wt:9s} largest |difference| ="
          f" {abs(mine - ref).max():.1e}")
m = MyKNNRegressor(5, "distance").fit(X, y).predict(X[:3])
print("on training rows the prediction is the own target:",
      np.allclose(m, y[:3]))
