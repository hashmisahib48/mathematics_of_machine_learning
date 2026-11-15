import numpy as np
from sklearn.linear_model import HuberRegressor


def scale(r, eps):
    """Solve n - k eps^2 = sum_in r^2 / s^2 for s (Eq. 13.3-8)."""
    lo, hi = 1e-12, 10 * np.abs(r).max() + 1.0
    for _ in range(100):                       # bisection
        s = (lo + hi) / 2
        out = np.abs(r) > eps * s
        g = len(r) - eps**2 * out.sum() - np.sum(r[~out]**2) / s**2
        lo, hi = (s, hi) if g < 0 else (lo, s)
    return s


class MyHuber:
    def __init__(self, epsilon=1.35, alpha=1e-4, n_iter=200):
        self.epsilon = epsilon
        self.alpha = alpha
        self.n_iter = n_iter

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        A = np.c_[np.ones(len(y)), X]          # ones column first
        P = np.diag([0.0] + [1.0] * X.shape[1])    # b not penalised
        th = np.linalg.lstsq(A, y, rcond=None)[0]  # OLS start
        eps = self.epsilon
        for _ in range(self.n_iter):
            r = y - A @ th
            sig = scale(r, eps)                    # Eq. 13.3-8
            # weights omega_i, Eq. 13.3-4
            om = np.minimum(1, eps * sig / (abs(r) + 1e-9))
            AO = A.T * om                          # Eq. 13.3-6
            new = np.linalg.solve(AO @ A + self.alpha * P, AO @ y)
            done = np.max(np.abs(new - th)) < 1e-12
            th = new
            if done:
                break
        self.intercept_, self.coef_ = th[0], th[1:]
        self.scale_ = scale(y - A @ th, eps)
        return self

    def predict(self, X):
        return np.asarray(X, float) @ self.coef_ + self.intercept_


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3))
    y = X @ [2.0, -1.0, 0.5] + 4 + rng.normal(size=100)
    y[:8] += 25                                # eight wild targets
    mine = MyHuber().fit(X, y)
    ref = HuberRegressor(max_iter=1000, tol=1e-9).fit(X, y)
    for name, m in (("mine", mine), ("sklearn", ref)):
        print(f"{name:8s}", np.round(m.coef_, 4),
              f"{m.intercept_:.4f}  scale {m.scale_:.4f}")
    assert np.allclose(mine.coef_, ref.coef_, atol=1e-3)
    assert np.allclose(mine.intercept_, ref.intercept_, atol=1e-3)
    assert abs(mine.scale_ - ref.scale_) < 1e-3
