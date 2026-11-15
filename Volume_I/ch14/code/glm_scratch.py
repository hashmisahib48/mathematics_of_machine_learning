# Tweedie GLM with a log link, fitted by Newton's method.
import numpy as np
from sklearn.linear_model import PoissonRegressor, GammaRegressor
from sklearn.linear_model import TweedieRegressor


class MyGLM:
    """power 1 = Poisson, 2 = gamma; objective Eq. 14.3-9."""

    def __init__(self, power=1.0, alpha=0.0, max_iter=50,
                 tol=1e-10):
        self.power, self.alpha = power, alpha
        self.max_iter, self.tol = max_iter, tol

    def fit(self, X, y, sample_weight=None):
        X, y = np.asarray(X, float), np.asarray(y, float)
        n, p = X.shape
        s = np.ones(n) if sample_weight is None else sample_weight
        s = np.asarray(s, float) / np.sum(s)     # weights add to 1
        Z = np.column_stack([X, np.ones(n)])       # intercept last
        pen = np.r_[np.full(p, self.alpha), 0.0]   # b is free
        beta = np.zeros(p + 1)
        beta[-1] = np.log(s @ y)                 # start: mu = mean
        q = self.power
        for self.n_iter_ in range(1, self.max_iter + 1):
            mu = np.exp(Z @ beta)                  # Eq. 14.3-1
            r = s * (y - mu) * mu ** (1 - q)
            g = -Z.T @ r + pen * beta              # Eq. 14.3-10
            if np.abs(g).max() < self.tol:
                break
            h = s * mu ** (1 - q) * (mu + (q - 1) * (y - mu))
            H = Z.T @ (h[:, None] * Z) + np.diag(pen)  # Eq. 14.3-11
            beta = beta - np.linalg.solve(H, g)    # Eq. 14.3-13
        self.coef_, self.intercept_ = beta[:-1], beta[-1]
        return self

    def predict(self, X):
        eta = np.asarray(X, float) @ self.coef_ + self.intercept_
        return np.exp(eta)
