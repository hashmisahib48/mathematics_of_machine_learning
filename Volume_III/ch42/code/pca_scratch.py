# PCA from scratch through the SVD of the centred data.
import numpy as np


class MyPCA:
    def __init__(self, n_components=2, whiten=False):
        self.n_components = n_components
        self.whiten = whiten

    def fit(self, X):
        X = np.asarray(X, float)
        n = len(X)
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        U, s, Vt = np.linalg.svd(Xc, full_matrices=False)  # 42.4-1
        big = np.abs(Vt).argmax(axis=1)         # svd_flip rule:
        rows = np.arange(len(Vt))               # largest entry of
        Vt = Vt * np.sign(Vt[rows, big])[:, None]   # a row > 0
        k = self.n_components
        self.components_ = Vt[:k]
        var = s ** 2 / (n - 1)                  # Eq. 42.4-2
        self.explained_variance_ = var[:k]
        self.explained_variance_ratio_ = var[:k] / var.sum()
        return self

    def transform(self, X):
        Z = (np.asarray(X, float) - self.mean_) @ self.components_.T
        if self.whiten:                         # Eq. 42.4-4
            Z = Z / np.sqrt(self.explained_variance_)
        return Z

    def inverse_transform(self, Z):
        if self.whiten:
            Z = Z * np.sqrt(self.explained_variance_)
        return Z @ self.components_ + self.mean_    # Eq. 42.4-3


# --- the test: agreement with scikit-learn on the digits data
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

X, y = load_digits(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)
for whiten in (False, True):
    mine = MyPCA(10, whiten).fit(Xtr)
    ref = PCA(10, whiten=whiten, svd_solver="full").fit(Xtr)
    for a in ("components_", "explained_variance_",
              "explained_variance_ratio_", "mean_"):
        assert np.allclose(getattr(mine, a), getattr(ref, a))
    Zm, Zr = mine.transform(Xte), ref.transform(Xte)
    assert np.allclose(Zm, Zr)
    assert np.allclose(mine.inverse_transform(Zm),
                       ref.inverse_transform(Zr))
    print("whiten", whiten, "agrees with PCA; max |dZ| =",
          f"{np.abs(Zm - Zr).max():.1e}")
print("variance ratio", mine.explained_variance_ratio_[:3].round(4))
