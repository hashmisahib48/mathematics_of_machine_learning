# Polynomial and B-spline feature maps from scratch.
import numpy as np
from itertools import combinations_with_replacement as cwr

class MyPoly:                   # monomials of total degree <= d
    def __init__(self, degree=2):
        self.degree = degree

    def fit(self, X, y=None):
        self.p_ = np.asarray(X).shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        cols = [np.prod(X[:, list(c)], axis=1)  # empty product = 1
                for k in range(self.degree + 1)  # Eq. 15.3-1
                for c in cwr(range(self.p_), k)]
        return np.column_stack(cols)

class MySpline:                 # B-splines of one feature
    def __init__(self, n_knots=5, degree=3):
        self.n_knots, self.degree = n_knots, degree

    def fit(self, x, y=None):
        x = np.asarray(x, dtype=float).ravel()
        d, n = self.degree, self.n_knots
        h = (x.max() - x.min()) / (n - 1)       # Eq. 15.4-3
        self.t_ = x.min() + h * np.arange(-d, n + d)
        return self

    def transform(self, x):
        d, t = self.degree, self.t_
        x = np.clip(np.ravel(x).astype(float), t[d], t[-d - 1])
        j = np.searchsorted(t, x, "right") - 1  # interval of x
        j = np.clip(j, d, len(t) - d - 2)
        B = np.zeros((len(x), len(t) - 1))
        B[np.arange(len(x)), j] = 1.0           # Eq. 15.4-1
        for k in range(1, d + 1):               # Eq. 15.4-2
            a = (x[:, None] - t[:-k-1]) / (t[k:-1] - t[:-k-1])
            c = (t[k+1:] - x[:, None]) / (t[k+1:] - t[1:-k])
            B = a * B[:, :-1] + c * B[:, 1:]
        return B
