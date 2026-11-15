# Binary logistic regression from scratch, by Newton's method.
import numpy as np
from sklearn.linear_model import LogisticRegression


class MyLogReg:
    """Minimises C * sum(log loss) + 0.5 * |w|^2 (Eq. 16.5-1)."""

    def __init__(self, C=1.0, max_iter=50):
        self.C, self.max_iter = C, max_iter

    def fit(self, X, y):
        A = np.c_[np.asarray(X, float), np.ones(len(y))]  # last: b
        reg = np.r_[np.ones(A.shape[1] - 1), 0.0]         # b free
        v = np.zeros(A.shape[1])
        for it in range(self.max_iter):
            p = 1 / (1 + np.exp(-A @ v))              # Eq. 16.3-1
            g = self.C * A.T @ (p - y) + reg * v      # Eq. 16.3-7
            if np.linalg.norm(g) < 1e-10:
                break
            S = p * (1 - p)
            H = self.C * A.T @ (A * S[:, None]) + np.diag(reg)
            v = v - np.linalg.solve(H, g)             # Eq. 16.3-10
        self.coef_, self.intercept_ = v[:-1], v[-1]
        self.n_iter_ = it
        return self

    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-(np.asarray(X) @ self.coef_
                              + self.intercept_)))
        return np.c_[1 - p, p]

rng = np.random.default_rng(1)
X = rng.normal(size=(200, 4))
y = (X @ [2.0, -1.0, 0.5, 0.0] + rng.normal(size=200) > 0) * 1
ref = LogisticRegression(C=0.5, tol=1e-12).fit(X, y)
m = MyLogReg(C=0.5).fit(X, y)
assert np.allclose(m.coef_, ref.coef_[0], atol=1e-5)
assert np.isclose(m.intercept_, ref.intercept_[0], atol=1e-5)
assert np.allclose(m.predict_proba(X), ref.predict_proba(X),
                   atol=1e-5)
for name, mod in (("MyLogReg", m), ("scikit-learn", ref)):
    print(f"{name:12s}", np.round(mod.coef_.ravel(), 4),
          round(float(np.ravel(mod.intercept_)[0]), 4),
          " iterations", np.ravel(mod.n_iter_)[0])
