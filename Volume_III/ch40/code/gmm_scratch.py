import numpy as np
from scipy.special import logsumexp


class MyGMM:
    """EM for a Gaussian mixture with full covariances."""
    def __init__(self, weights, means, covs, tol=1e-3,
                 max_iter=100, reg_covar=1e-6):
        self.pi_, self.mu_ = np.array(weights), np.array(means)
        self.cov_ = np.array(covs)          # the initial guess
        self.tol, self.max_iter, self.reg = tol, max_iter, reg_covar

    def _e_step(self, X):
        n, p = X.shape
        lj = np.empty((n, len(self.pi_)))   # ln pi_k N(x_i | k)
        for k, (m, S) in enumerate(zip(self.mu_, self.cov_)):
            d = X - m
            q = np.sum(d * np.linalg.solve(S, d.T).T, axis=1)
            lj[:, k] = np.log(self.pi_[k]) - 0.5 * (
                p * np.log(2 * np.pi) + np.linalg.slogdet(S)[1] + q)
        lse = logsumexp(lj, axis=1)         # ln p(x_i), Eq. 40.3-3
        return lse, np.exp(lj - lse[:, None])   # Eq. 40.3-4

    def fit(self, X):
        n, p = X.shape
        prev = -np.inf
        for it in range(1, self.max_iter + 1):
            lse, R = self._e_step(X)
            Nk = R.sum(axis=0)           # M-step, Eqs. 40.4-5 to 7
            self.pi_ = Nk / n
            self.mu_ = R.T @ X / Nk[:, None]
            for k in range(len(Nk)):
                d = X - self.mu_[k]
                self.cov_[k] = (R[:, k] * d.T) @ d / Nk[k]
                self.cov_[k] += self.reg * np.eye(p)
            self.n_iter_, self.lower_bound_ = it, lse.mean()
            if abs(self.lower_bound_ - prev) < self.tol:
                break
            prev = self.lower_bound_
        return self

    def predict_proba(self, X):
        return self._e_step(X)[1]
    def score_samples(self, X):
        return self._e_step(X)[0]
