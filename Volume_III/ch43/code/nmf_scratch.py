# NMF by multiplicative updates (Frobenius loss)
import numpy as np

EPS = np.finfo(np.float64).eps


class MyNMF:
    def __init__(self, n_components=2, max_iter=100):
        self.n_components = n_components
        self.max_iter = max_iter

    def _sweeps(self, X, W, H, update_H):
        for _ in range(self.max_iter):
            den = W @ (H @ H.T)                  # Eq. 43.3-6
            W *= (X @ H.T) / np.maximum(den, EPS)
            if update_H:
                den = (W.T @ W) @ H
                H *= (W.T @ X) / np.maximum(den, EPS)
        return W, H

    def fit_transform(self, X, W, H):
        X = np.asarray(X, float)
        W, H = self._sweeps(X, W.copy(), H.copy(), True)
        self.components_ = H
        self.reconstruction_err_ = np.linalg.norm(X - W @ H)
        return W

    def transform(self, X):                  # H fixed, only W moves
        X = np.asarray(X, float)
        k = len(self.components_)
        W = np.full((len(X), k), np.sqrt(X.mean() / k))
        return self._sweeps(X, W, self.components_, False)[0]


# --- the test: same start and updates as scikit-learn
from sklearn.datasets import load_digits
from sklearn.decomposition import NMF
from sklearn.model_selection import train_test_split

X, y = load_digits(return_X_y=True)
Xtr, Xte = train_test_split(X, test_size=0.25, random_state=0,
                            stratify=y)
rng = np.random.RandomState(0)
W0 = rng.uniform(0.5, 1.5, (len(Xtr), 10))
H0 = rng.uniform(0.5, 1.5, (10, 64))
mine = MyNMF(10, 100)
Wm = mine.fit_transform(Xtr, W0, H0)
ref = NMF(10, solver="mu", init="custom", max_iter=100, tol=0.0)
Wr = ref.fit_transform(Xtr, W=W0.copy(), H=H0.copy())
print("error: mine", mine.reconstruction_err_.round(6),
      " sklearn", ref.reconstruction_err_.round(6))
assert np.allclose(Wm, Wr)
assert np.allclose(mine.components_, ref.components_)
assert np.allclose(mine.transform(Xte), ref.transform(Xte))
print("W, H and transform agree; max |dH| =",
      f"{np.abs(mine.components_ - ref.components_).max():.1e}")
