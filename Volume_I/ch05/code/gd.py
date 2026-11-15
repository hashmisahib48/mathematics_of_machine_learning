# Gradient descent on the delivery loss, raw and rescaled features.
import numpy as np

x = np.arange(1.0, 6.0)
y = np.array([3.0, 4.0, 6.0, 7.0, 10.0])


def descend(X, eta, tol=1e-6, max_iter=100000):
    w = np.zeros(X.shape[1])
    for t in range(max_iter):
        g = -2 * X.T @ (y - X @ w)            # gradient, Eq. 5.3-3
        if np.linalg.norm(g) < tol:
            return w, t                       # stopping rule
        w = w - eta * g                       # update, Eq. 5.3-1
        if np.linalg.norm(w) > 1e6:
            return w, None                    # diverged
    return w, None


s = np.sqrt(2)                                # std of x is sqrt(2)
designs = {"raw x": np.c_[np.ones(5), x],
           "centred x": np.c_[np.ones(5), x - 3],
           "standardised x": np.c_[np.ones(5), (x - 3) / s]}
for name, X in designs.items():
    lam = np.linalg.eigvalsh(2 * X.T @ X)     # curvatures
    eta = 1 / lam[-1]                         # step 1 / lambda_max
    w, n = descend(X, eta)
    print(f"{name:15s} cond {lam[-1] / lam[0]:5.1f}  eta {eta:.4f}"
          f"  steps {n:4d}  w {w.round(4)}")

X = designs["raw x"]
lam = np.linalg.eigvalsh(2 * X.T @ X)
print("eigenvalues", lam.round(2), " limit", (2 / lam[-1]).round(5))
for eta in (0.002, 0.01, 0.0165, 0.0172):
    w, n = descend(X, eta)
    print(f"eta {eta:.4f}: steps {n}")
