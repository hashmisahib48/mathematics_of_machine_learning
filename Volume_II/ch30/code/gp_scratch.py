# Gaussian process regression with a fixed RBF kernel, from scratch.
import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular


def rbf(A, B, sf2, ell):
    d2 = ((A[:, None, :] - B[None, :, :])**2).sum(-1)
    return sf2 * np.exp(-d2 / (2 * ell**2))          # Eq. 30.5-1


class MyGP:
    def __init__(self, sf2=1.0, ell=1.0, noise=1e-10):
        self.sf2, self.ell, self.noise = sf2, ell, noise

    def fit(self, X, y):
        self.X_, self.y_ = X, y
        S = rbf(X, X, self.sf2, self.ell)
        S[np.diag_indices_from(S)] += self.noise     # K + s2 I
        self.L_ = np.linalg.cholesky(S)              # S = L L^T
        self.alpha_ = cho_solve((self.L_, True), y)  # S^-1 y
        n = len(y)
        self.lml_ = (-0.5 * y @ self.alpha_          # Eq. 30.4-2
                     - np.log(np.diag(self.L_)).sum()
                     - 0.5 * n * np.log(2 * np.pi))
        return self

    def predict(self, Xs, return_std=False, return_cov=False):
        Ks = rbf(Xs, self.X_, self.sf2, self.ell)    # Eq. 30.3-8
        mean = Ks @ self.alpha_                      # Eq. 30.3-7
        V = solve_triangular(self.L_, Ks.T, lower=True)
        if return_cov:
            return mean, rbf(Xs, Xs, self.sf2, self.ell) - V.T @ V
        if return_std:
            var = self.sf2 - (V**2).sum(0)           # k** - v^T v
            return mean, np.sqrt(np.maximum(var, 0))
        return mean
