# Kernel ridge regression from scratch (no intercept).
import numpy as np
from sklearn.kernel_ridge import KernelRidge


class MyKRR:
    """Kernels: linear, poly, rbf, laplacian; gamma=None means
    1/n_features, as in sklearn.metrics.pairwise."""

    def __init__(self, alpha=1.0, kernel="rbf", gamma=None,
                 degree=3, coef0=1.0):
        self.alpha, self.kernel, self.gamma = alpha, kernel, gamma
        self.degree, self.coef0 = degree, coef0

    def _k(self, A, B):                    # Gram matrix of A and B
        g = 1.0 / A.shape[1] if self.gamma is None else self.gamma
        dot = A @ B.T
        if self.kernel == "linear":
            return dot
        if self.kernel == "poly":
            return (g * dot + self.coef0) ** self.degree
        if self.kernel == "laplacian":
            d1 = np.abs(A[:, None, :] - B[None, :, :]).sum(axis=2)
            return np.exp(-g * d1)
        sq = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1) - 2 * dot
        return np.exp(-g * np.maximum(sq, 0.0))      # Eq. 21.3-7

    def fit(self, X, y):
        self.X_fit_ = np.asarray(X, float)
        K = self._k(self.X_fit_, self.X_fit_)
        I = np.eye(len(K))
        self.dual_coef_ = np.linalg.solve(K + self.alpha * I, y)
        return self                                  # Eq. 21.4-3

    def predict(self, X):                            # Eq. 21.4-4
        K = self._k(np.asarray(X, float), self.X_fit_)
        return K @ self.dual_coef_


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3))
    y = np.sin(X[:, 0]) + X[:, 1] ** 2 + 0.1 * rng.normal(size=60)
    X_new = rng.normal(size=(5, 3))
    for kern in ("linear", "poly", "rbf", "laplacian"):
        kw = dict(alpha=0.5, kernel=kern, gamma=0.3)
        ours = MyKRR(**kw).fit(X, y)
        ref = KernelRidge(**kw).fit(X, y)
        assert np.allclose(ours.dual_coef_, ref.dual_coef_)
        assert np.allclose(ours.predict(X_new), ref.predict(X_new))
        err = abs(ours.predict(X_new) - ref.predict(X_new)).max()
        print(f"{kern:9s} matches KernelRidge, max diff {err:.1e}")
