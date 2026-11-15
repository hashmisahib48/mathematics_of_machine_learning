# Kernel density estimate from scratch, tested on scikit-learn.
import numpy as np
from scipy.special import gammaln, logsumexp
from sklearn.neighbors import KernelDensity


class MyKDE:
    def __init__(self, bandwidth=1.0, kernel="gaussian"):
        self.bandwidth, self.kernel = bandwidth, kernel

    def fit(self, X):
        self.X_ = np.asarray(X, float)
        return self

    def score_samples(self, X):               # ln of Eq. 47.3-2
        n, p = self.X_.shape
        h = self.bandwidth
        d = np.asarray(X, float)[:, None, :] - self.X_[None]
        u2 = (d ** 2).sum(-1) / h ** 2        # |u|^2
        logV = p / 2 * np.log(np.pi) - gammaln(p / 2 + 1)  # ball
        with np.errstate(divide="ignore"):
            if self.kernel == "gaussian":
                logk, logc = -u2 / 2, -p / 2 * np.log(2 * np.pi)
            elif self.kernel == "tophat":
                logk, logc = np.where(u2 < 1, 0.0, -np.inf), -logV
            else:                             # epanechnikov
                logk = np.log(np.clip(1 - u2, 0, None))
                logc = np.log((p + 2) / 2) - logV
        return (logsumexp(logk, axis=1) + logc
                - np.log(n) - p * np.log(h))


rng = np.random.default_rng(0)
X, Q = rng.normal(size=(60, 3)), rng.normal(size=(20, 3))
for kernel in ["gaussian", "tophat", "epanechnikov"]:
    mine = MyKDE(0.9, kernel).fit(X).score_samples(Q)
    ref = KernelDensity(bandwidth=0.9, kernel=kernel).fit(X)
    ok = np.allclose(mine, ref.score_samples(Q))
    print(f"{kernel:13s} agree: {ok}   first: {mine[0]:.6f}")
