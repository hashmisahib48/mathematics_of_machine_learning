# Gaussian kernel density estimate from scratch, tested on sklearn.
import numpy as np
from scipy.special import logsumexp
from sklearn.neighbors import KernelDensity


class MyKDE:
    def __init__(self, bandwidth=1.0):
        self.bandwidth = bandwidth

    def fit(self, X):
        self.X_ = np.asarray(X, float)
        return self

    def score_samples(self, X):               # ln f, Eq. 47.3-2
        n, p = self.X_.shape
        h = self.bandwidth
        d = np.asarray(X, float)[:, None, :] - self.X_[None]
        logk = -(d ** 2).sum(-1) / (2 * h ** 2)   # ln k(u)
        logc = -p / 2 * np.log(2 * np.pi)         # ln c_K
        return (logsumexp(logk, axis=1) + logc
                - np.log(n) - p * np.log(h))


X, Q = np.random.default_rng(0).normal(size=(2, 60, 3))
ref = KernelDensity(bandwidth=0.9).fit(X)
diff = MyKDE(0.9).fit(X).score_samples(Q) - ref.score_samples(Q)
print("largest difference in ln f:", np.abs(diff).max())
