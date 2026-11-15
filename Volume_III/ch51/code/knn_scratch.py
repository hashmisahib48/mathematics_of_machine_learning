import numpy as np
from sklearn.impute import KNNImputer
from holes import split, punch

class MyKNNImputer:
    def __init__(self, k=5, weights="uniform"):
        self.k, self.weights = k, weights
    def fit(self, X, y=None):
        self.X_ = np.asarray(X, float)
        return self
    def _dist(self, x):                  # Eq. 51.4-1 to every row
        Z = self.X_
        S = ~np.isnan(x) & ~np.isnan(Z)          # present in both
        diff = np.nan_to_num(x) - np.nan_to_num(Z)
        with np.errstate(divide="ignore", invalid="ignore"):
            sq = np.where(S, diff ** 2, 0).sum(1)
            d2 = len(x) / S.sum(1) * sq
        return np.sqrt(d2)                       # nan if S is empty
    def transform(self, X):
        X = np.array(X, float)
        for i in np.flatnonzero(np.isnan(X).any(1)):
            d = self._dist(X[i])
            for j in np.flatnonzero(np.isnan(X[i])):
                ok = np.flatnonzero(~np.isnan(self.X_[:, j] + d))
                if len(ok) == 0:               # no usable donor
                    X[i, j] = np.nanmean(self.X_[:, j]); continue
                idx = ok[np.argsort(d[ok], kind="stable")[:self.k]]
                w = 1 / d[idx] if self.weights == "distance" \
                    else np.ones(len(idx))     # Eq. 51.4-2
                X[i, j] = w @ self.X_[idx, j] / w.sum()
        return X

Xtr, Xte, ytr, yte = split()
A, B = punch(Xtr, "MCAR", 0.2, 0), punch(Xte, "MCAR", 0.2, 1)
for w in ("uniform", "distance"):
    mine = MyKNNImputer(5, w).fit(A)
    ref = KNNImputer(n_neighbors=5, weights=w).fit(A)
    err = np.abs(mine.transform(B) - ref.transform(B)).max()
    print(f"weights={w:8s} max |difference| = {err:.1e}")
    assert err < 1e-10
