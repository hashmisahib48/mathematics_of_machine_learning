# Problem 5B.1: momentum against plain gradient descent.
import numpy as np

x = np.arange(1.0, 6.0)
y = np.array([3.0, 4.0, 6.0, 7.0, 10.0])
X = np.c_[np.ones(5), x]
J = lambda w: ((y - X @ w) ** 2).sum()           # loss
grad = lambda w: -2 * X.T @ (y - X @ w)          # Eq. 5.3-3
lam = np.linalg.eigvalsh(2 * X.T @ X)            # curvatures
print("eigenvalues", lam.round(2))


def steps_to_target(eta, mu, excess):
    w, v = np.zeros(2), np.zeros(2)       # velocity starts at 0
    for t in range(5000):
        if J(w) < 1.10 + excess:                 # minimum is 1.10
            return t
        v = mu * v - eta * grad(w)               # Eq. 5.4-5
        w = w + v
    return None


for excess in (0.01, 1e-6):
    plain = steps_to_target(1 / lam[-1], 0.0, excess)  # mu = 0
    heavy = steps_to_target(0.008, 0.8, excess)
    print(f"within {excess:g} of minimum: plain {plain} steps, "
          f"momentum {heavy} steps")

# why: shrink factor per step in each eigen-direction
runs = (("plain", 1 / lam[-1], 0.0), ("momentum", 0.008, 0.8))
for name, eta, mu in runs:
    for h in lam:    # error recursion: z^2 - b z + mu = 0
        b = 1 + mu - eta * h
        z = np.roots([1, -b, mu])
        print(f"{name:8s} h={h:7.2f}: b^2={b * b:.3f},"
              f" 4mu={4 * mu:.3f}, factor {abs(z).max():.3f}")
