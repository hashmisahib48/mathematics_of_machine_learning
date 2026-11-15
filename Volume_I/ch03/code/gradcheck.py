# Checking an analytic gradient against central differences.
import numpy as np

def num_grad(f, w, h=1e-6):
    g = np.zeros_like(w)
    for j in range(w.size):
        e = np.zeros_like(w); e[j] = h
        g[j] = (f(w + e) - f(w - e)) / (2 * h)
    return g

def rel_error(g1, g2):
    return np.linalg.norm(g1 - g2) / (np.linalg.norm(g1)
                                      + np.linalg.norm(g2))

rng = np.random.default_rng(0)
X = rng.normal(size=(20, 3)); y = rng.normal(size=20)
w = rng.normal(size=3); alpha = 0.5

J = lambda w: np.sum((y - X @ w) ** 2)              # Eq. 3.7-5
grad = lambda w: -2 * X.T @ (y - X @ w)             # Eq. 3.7-6
ridge = lambda w: J(w) + alpha * w @ w
ridge_grad = lambda w: grad(w) + 2 * alpha * w
wrong = lambda w: -X.T @ (y - X @ w)                # forgot the 2

for name, f, g in [("least squares", J, grad),
                   ("ridge", ridge, ridge_grad),
                   ("wrong gradient", J, wrong)]:
    e = rel_error(g(w), num_grad(f, w))
    print(f"{name:<15} relative error = {e:.1e}",
          " OK" if e < 1e-6 else " <-- BUG")

print("\nstep h    forward error  central error  (exp at x = 1)")
for k in [2, 4, 6, 8, 10, 12]:
    h = 10.0 ** -k
    fwd = (np.exp(1 + h) - np.exp(1)) / h
    cen = (np.exp(1 + h) - np.exp(1 - h)) / (2 * h)
    ef, ec = abs(fwd - np.e), abs(cen - np.e)
    print(f"1e-{k:<2d} {ef:14.2e} {ec:14.2e}")
