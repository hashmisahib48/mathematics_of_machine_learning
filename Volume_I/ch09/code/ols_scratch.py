# Ordinary least squares from scratch, three ways.
import numpy as np
from sklearn.linear_model import LinearRegression


class MyOLS:
    """OLS with an intercept; method is 'normal', 'qr' or 'svd'."""

    def __init__(self, method="svd"):
        self.method = method

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        x_mean, y_mean = X.mean(axis=0), y.mean()
        Xc, yc = X - x_mean, y - y_mean        # centre: Eq. 9.4-8
        if self.method == "normal":            # Eq. 9.4-5
            w = np.linalg.solve(Xc.T @ Xc, Xc.T @ yc)
        elif self.method == "qr":              # Eq. 9.4-9
            Q, R = np.linalg.qr(Xc)
            w = np.linalg.solve(R, Q.T @ yc)
        else:                                  # Eq. 9.4-10
            U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
            tol = s.max() * max(Xc.shape) * np.finfo(float).eps
            keep = s > tol          # drop zero directions
            w = Vt[keep].T @ ((U[:, keep].T @ yc) / s[keep])
        self.coef_ = w
        self.intercept_ = y_mean - x_mean @ w
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    w_true = np.array([2.0, -1.0, 0.5, 3.0])
    y = X @ w_true + 4.0 + rng.normal(size=200)
    ref = LinearRegression().fit(X, y)
    for method in ("normal", "qr", "svd"):
        m = MyOLS(method).fit(X, y)
        assert np.allclose(m.coef_, ref.coef_)
        assert np.allclose(m.intercept_, ref.intercept_)
        print(f"{method:6s} matches scikit-learn:",
              np.round(m.coef_, 4), round(m.intercept_, 4))
