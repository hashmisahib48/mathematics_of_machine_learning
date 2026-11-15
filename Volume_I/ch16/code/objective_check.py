# scikit-learn's objective (Eq. 16.5-1) and its optimality test.
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

X, y = make_classification(n_samples=200, n_features=5,
                           n_informative=3, n_redundant=0,
                           random_state=0)
X = StandardScaler().fit_transform(X)
s = 2.0 * y - 1                               # labels -1, +1


def J(w, b, C, rho):                          # Eq. 16.5-1
    loss = np.logaddexp(0, -s * (X @ w + b)).sum()
    if np.isinf(C):
        return loss
    return C * loss + rho * abs(w).sum() + (1 - rho) / 2 * w @ w


def violation(w, b, C, rho):                  # Eq. 16.5-4
    p = 1 / (1 + np.exp(-(X @ w + b)))
    pen = not np.isinf(C)
    c = C if pen else 1.0
    l1, l2 = (rho, 1 - rho) if pen else (0.0, 0.0)
    g = c * X.T @ (p - y) + l2 * w
    v = np.where(w != 0, abs(g + l1 * np.sign(w)),
                 np.maximum(abs(g) - l1, 0))
    return v.max(), abs(c * (p - y).sum())


print("case     solver                  J  weights   interc zeros")
cases = [("L2", dict(C=0.1, l1_ratio=0), "lbfgs"),
         ("L1", dict(C=0.1, l1_ratio=1), "saga"),
         ("elastic", dict(C=0.1, l1_ratio=0.3), "saga"),
         ("none", dict(C=np.inf), "newton-cholesky"),
         ("L1", dict(C=0.1, l1_ratio=1), "liblinear")]
for name, kw, solver in cases:
    m = LogisticRegression(solver=solver, tol=1e-12, max_iter=50000,
                           random_state=0, **kw).fit(X, y)
    w, b = m.coef_.ravel(), m.intercept_[0]
    C, rho = kw["C"], kw.get("l1_ratio", 0)
    v1, v2 = violation(w, b, C, rho)
    print(f"{name:8s} {solver:15s} {J(w, b, C, rho):8.4f} {v1:8.1e}"
          f" {v2:8.1e} {int((w == 0).sum()):5d}")
