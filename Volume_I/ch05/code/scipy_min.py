# One convex objective, four optimisers, and scikit-learn's answer.
import numpy as np
from scipy.optimize import minimize
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
X = StandardScaler().fit_transform(X)
s = 2.0 * y - 1                                # labels -1, +1
A = np.c_[X, np.ones(len(y))]                  # last: intercept
C = 1.0


def J(v):                                      # Eq. 5.5-4
    loss = np.logaddexp(0, -s * (A @ v)).sum()
    return 0.5 * v[:-1] @ v[:-1] + C * loss


def grad(v):
    g = -C * A.T @ (s / (1 + np.exp(s * (A @ v))))
    g[:-1] += v[:-1]
    return g


ref = LogisticRegression(C=C, tol=1e-10, max_iter=1000).fit(X, y)
v_ref = np.r_[ref.coef_[0], ref.intercept_]
print(f"scikit-learn: J = {J(v_ref):.6f}  n_iter_ {ref.n_iter_[0]}")
for m in ("CG", "BFGS", "L-BFGS-B", "Newton-CG"):
    r = minimize(J, np.zeros(31), jac=grad, method=m, tol=1e-10)
    d = np.abs(r.x - v_ref).max()
    print(f"{m:10s} J = {r.fun:.6f}  nit {r.nit:3d}  "
          f"nfev {r.nfev:3d}  max|dw| {d:.1e}")
