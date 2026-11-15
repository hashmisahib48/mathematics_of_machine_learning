# Chord test for convexity, and the curvature of two losses.
import numpy as np

fs = {
    "squared r^2": lambda r: r**2,
    "absolute |r|": np.abs,
    "log loss": lambda z: np.logaddexp(0, -z),    # ln(1 + e^-z)
    "sin z": np.sin,
    "z^4 - 3 z^2": lambda z: z**4 - 3 * z**2,
}
rng = np.random.default_rng(0)
a, b, t = (rng.uniform(lo, hi, 100000)
           for lo, hi in [(-3, 3), (-3, 3), (0, 1)])
for name, f in fs.items():
    gap = t * f(a) + (1 - t) * f(b) - f(t * a + (1 - t) * b)
    print(f"{name:13s} smallest gap {gap.min():+.2e}   "
          f"violations {np.sum(gap < -1e-12):6d}")

# Curvature: Hessian of the squared loss is 2 X^T X (Chapter 3).
def eigs(X):
    return np.round(np.linalg.eigvalsh(2 * X.T @ X), 2) + 0.0


x = np.arange(1.0, 6.0)
X = np.c_[np.ones(5), x]
print("delivery eigenvalues:", eigs(X))
print("with 2x added:       ", eigs(np.c_[X, 2 * x]))
z = np.array([-4.0, -2, 0, 2, 4])
s = 1 / (1 + np.exp(-z))
print("log-loss curvature s(1-s):", np.round(s * (1 - s), 4))
