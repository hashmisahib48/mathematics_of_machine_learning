# Ridge regression from scratch, two ways.
import numpy as np
from sklearn.linear_model import Ridge


class MyRidge:
    """min ||y - Xw - b||^2 + alpha ||w||^2; b is not penalised."""

    def __init__(self, alpha=1.0, method="cholesky"):
        self.alpha = alpha
        self.method = method

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        x_mean, y_mean = X.mean(axis=0), y.mean()
        Xc, yc = X - x_mean, y - y_mean      # Eq. 10.3-3
        U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        f = s**2 / (s**2 + self.alpha)       # filter factors
        self.df_ = f.sum()                   # Eq. 10.4-6
        if self.method == "cholesky":        # Eq. 10.3-5
            A = Xc.T @ Xc + self.alpha * np.eye(X.shape[1])
            w = np.linalg.solve(A, Xc.T @ yc)
        else:                                # Eq. 10.4-3
            w = Vt.T @ (f / s * (U.T @ yc))
        self.coef_ = w
        self.intercept_ = y_mean - x_mean @ w
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 5))
    y = X @ [2.0, -1.0, 0.5, 3.0, 0.0] + 4 + rng.normal(size=100)
    for a in (0.1, 10.0, 1000.0):
        ref = Ridge(alpha=a).fit(X, y)
        for method in ("cholesky", "svd"):
            m = MyRidge(a, method).fit(X, y)
            assert np.allclose(m.coef_, ref.coef_)
            assert np.allclose(m.intercept_, ref.intercept_)
        print(f"alpha={a:6.1f}  df={m.df_:.3f}  w1..w3 =",
              np.round(m.coef_[:3], 3))
    Xd = np.c_[X[:, 0], X[:, 0], X[:, 1:]]     # a duplicated column
    ref = Ridge(alpha=1.0).fit(Xd, y)
    m = MyRidge(1.0, "cholesky").fit(Xd, y)
    assert np.allclose(m.coef_, ref.coef_)
    print("duplicated column:", np.round(m.coef_[:2], 4))
