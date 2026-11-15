import numpy as np


class MyBayesRidge:
    """Bayesian ridge with MacKay's evidence updates."""

    def __init__(self, tol=1e-3, max_iter=300, fixed=None):
        self.tol, self.max_iter = tol, max_iter
        self.fixed = fixed          # (alpha, lambda) or None

    def _post(self, X, y, alpha, lam):
        A = lam * np.eye(X.shape[1]) + alpha * X.T @ X # Eq. 12.3-7
        S = np.linalg.inv(A)                           # Eq. 12.3-11
        return alpha * S @ X.T @ y, S

    def fit(self, X, y):
        self.xbar_, self.ybar_ = X.mean(0), y.mean()
        Xc, yc = X - self.xbar_, y - self.ybar_
        n, old = len(yc), None
        d2 = np.linalg.svd(Xc, compute_uv=False) ** 2
        alpha, lam = self.fixed or (1 / yc.var(), 1.0)
        for _ in range(self.max_iter):
            m, S = self._post(Xc, yc, alpha, lam)
            if self.fixed:
                break
            gamma = np.sum(alpha * d2 / (lam + alpha * d2))
            sse = np.sum((yc - Xc @ m) ** 2)
            lam = gamma / (m @ m)                       # Eq. 12.4-8
            alpha = (n - gamma) / sse
            if old is not None and abs(old - m).sum() < self.tol:
                break
            old = m.copy()
        self.alpha_, self.lambda_ = alpha, lam
        self.coef_, self.sigma_ = self._post(Xc, yc, alpha, lam)
        self.intercept_ = self.ybar_ - self.xbar_ @ self.coef_
        return self

    def predict(self, X, return_std=False):
        Xc = X - self.xbar_
        mean = Xc @ self.coef_ + self.ybar_
        if not return_std:
            return mean
        var = 1 / self.alpha_ + np.sum(Xc @ self.sigma_ * Xc, 1)
        return mean, np.sqrt(var)                       # Eq. 12.4-2
