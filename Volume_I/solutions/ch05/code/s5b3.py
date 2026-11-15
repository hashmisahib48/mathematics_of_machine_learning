# Problem 5B.3: Nelder-Mead (values only) against BFGS (gradient).
import numpy as np
from scipy.optimize import minimize

x = np.arange(1.0, 6.0)
y = np.array([3.0, 4.0, 6.0, 7.0, 10.0])
X = np.c_[np.ones(5), x]
J = lambda w: float(((y - X @ w) ** 2).sum())    # delivery loss
grad = lambda w: -2 * X.T @ (y - X @ w)          # Eq. 5.3-3

nm = minimize(J, [0, 0], method="Nelder-Mead",
              options={"xatol": 1e-8, "fatol": 1e-10})
bf = minimize(J, [0, 0], jac=grad, method="BFGS")

for name, r in (("Nelder-Mead", nm), ("BFGS", bf)):
    print(f"{name:11s} x={r.x.round(4)} J={r.fun:.6f} nit={r.nit:3d}"
          f" nfev={r.nfev:3d} njev={getattr(r, 'njev', 0)}")
print("exact solution", np.linalg.solve(X.T @ X, X.T @ y).round(4))

# BFGS without the gradient: finite differences cost extra evaluations
fd = minimize(J, [0, 0], method="BFGS")
print(f"BFGS, numerical gradient: nfev={fd.nfev}, njev={fd.njev}")
