# Support vector regression from scratch: the dual of Eq. 23.3-4.
import numpy as np
from scipy.optimize import minimize


class MySVR:
    def __init__(self, C=1.0, epsilon=0.1, kernel="rbf", gamma=0.5):
        self.C, self.epsilon = C, epsilon
        self.kernel, self.gamma = kernel, gamma

    def _k(self, A, B):                  # kernels of Chapter 21
        G = A @ B.T
        if self.kernel == "linear":
            return G
        if self.kernel == "poly":        # degree 3, coef0 = 1
            return (self.gamma * G + 1.0) ** 3
        d2 = (A**2).sum(1)[:, None] + (B**2).sum(1) - 2 * G
        return np.exp(-self.gamma * d2)  # rbf

    def fit(self, X, y):
        n, C, eps = len(y), self.C, self.epsilon
        K, t = self._k(X, X), 1e-6 * C
        def neg_dual(a):                 # minus Eq. 23.3-4
            beta = a[:n] - a[n:]         # a = (alpha, alpha*)
            g = K @ beta
            return (0.5 * beta @ g + eps * a.sum() - y @ beta,
                    np.r_[g + eps - y, -g + eps + y])
        eq = {"type": "eq",              # sum of beta = 0
              "fun": lambda a: a[:n].sum() - a[n:].sum()}
        res = minimize(neg_dual, np.zeros(2 * n), jac=True,
                       method="SLSQP", constraints=eq,
                       bounds=[(0, C)] * (2 * n), tol=1e-12)
        al, als = res.x[:n], res.x[n:]
        beta, s = al - als, K @ (al - als)
        up = (al > t) & (al < C - t)     # free, above the tube
        lo = (als > t) & (als < C - t)   # free, below the tube
        b = np.r_[(y - eps - s)[up], (y + eps - s)[lo]]  # 23.3-7
        sv = np.abs(beta) > t
        self.support_, self.dual_coef_ = np.where(sv)[0], beta[sv]
        self.support_vectors_, self.intercept_ = X[sv], b.mean()
        return self

    def predict(self, X):                # Eq. 23.3-6
        K = self._k(X, self.support_vectors_)
        return K @ self.dual_coef_ + self.intercept_


if __name__ == "__main__":               # test against scikit-learn
    from sklearn.svm import SVR
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3))
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1]**2
    y += 0.1 * rng.normal(size=60)
    Xt = rng.normal(size=(200, 3))
    for kern, gam in [("linear", 1.0), ("rbf", 0.5), ("poly", 0.3)]:
        a = MySVR(1.0, 0.1, kern, gam).fit(X, y)
        ref = SVR(C=1.0, epsilon=0.1, kernel=kern, gamma=gam,
                  coef0=1.0, tol=1e-8).fit(X, y)
        assert np.array_equal(a.support_, ref.support_)
        d = np.abs(a.predict(Xt) - ref.predict(Xt)).max()
        e = np.abs(a.dual_coef_ - ref.dual_coef_[0]).max()
        assert d < 1e-4 and e < 1e-4
        print(f"{kern:6s} SV {len(a.support_):2d} df {d:.1e} "
              f"dbeta {e:.1e} b {a.intercept_:.5f}"
              f"/{ref.intercept_[0]:.5f}")
