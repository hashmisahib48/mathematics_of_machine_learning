# Two Lloyd iterations on the eight customers (Example 36.7-1).
import numpy as np

X = np.array([[3, 7], [1, 3], [1, 4], [3, 8],
              [4, 0], [2, 1], [5, 8], [1, 8]], float)
mu = np.array([[0, 5], [5, 5]], float)       # the starting depots
np.set_printoptions(precision=4, suppress=True)

old = None
for it in (1, 2, 3):
    diff = X[:, None, :] - mu[None, :, :]    # shape (8, 2, 2)
    d2 = (diff ** 2).sum(axis=2)             # squared distances
    c = d2.argmin(axis=1)                    # Eq. 36.4-1
    J = d2[np.arange(8), c].sum()            # Eq. 36.3-1
    print(f"iteration {it}: depots {mu.tolist()}")
    print("  d2 to depot 1:", d2[:, 0])
    print("  d2 to depot 2:", d2[:, 1])
    print("  labels:", c + 1, f" inertia {J:.4f}")
    if old is not None and np.array_equal(c, old):
        print("  labels unchanged: converged")
        break
    old = c
    mu = np.array([X[c == k].mean(axis=0) for k in (0, 1)])  # 36.4-2
