# Linear discriminant analysis from scratch (Listing 18.2).
import numpy as np
from scipy.linalg import eigh


class MyLDA:
    """LDA, fixed shrinkage a; Eqs. 18.3-10, 18.5-3."""

    def __init__(self, shrinkage=0.0):
        self.shrinkage = shrinkage

    def fit(self, X, y):
        cl = self.classes_ = np.unique(y)
        pri = self.priors_ = np.array([np.mean(y == k) for k in cl])
        mu = self.means_ = np.array([X[y == k].mean(0) for k in cl])
        Sw = sum(p * np.cov(X[y == k].T, bias=True)  # divide by n_k
                 for p, k in zip(pri, cl))
        a, p = self.shrinkage, X.shape[1]
        Sw = (1 - a) * Sw + a * np.trace(Sw) / p * np.eye(p)
        self.covariance_ = Sw                         # Eq. 18.3-10
        self.coef_ = np.linalg.solve(Sw, mu.T).T      # Eq. 18.3-5
        self.intercept_ = (-0.5 * np.sum(mu * self.coef_, axis=1)
                           + np.log(pri))
        self.xbar_ = pri @ mu
        Sb = np.cov(mu.T, aweights=pri, bias=True)    # between
        lam, V = eigh(Sb, Sw)                         # Eq. 18.4-4
        self.scalings_ = V[:, ::-1]
        ratio = lam[::-1] / lam.sum()
        self.explained_variance_ratio_ = ratio[:len(cl) - 1]
        return self

    def decision_function(self, X):
        return X @ self.coef_.T + self.intercept_     # Eq. 18.3-5

    def predict_proba(self, X):
        z = self.decision_function(X)
        e = np.exp(z - z.max(axis=1, keepdims=True))  # Eq. 18.3-6
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.decision_function(X).argmax(1)]

    def transform(self, X, k=2):
        return (X - self.xbar_) @ self.scalings_[:, :k]
