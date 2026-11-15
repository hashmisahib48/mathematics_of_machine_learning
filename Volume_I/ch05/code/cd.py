# Coordinate descent for least squares on the delivery data.
import numpy as np

x = np.arange(1.0, 6.0)
y = np.array([3.0, 4.0, 6.0, 7.0, 10.0])


def sweeps(X, tol=1e-8, show=0):
    w = np.zeros(X.shape[1])
    for k in range(1, 100000):
        old = w.copy()
        for j in range(X.shape[1]):           # one coordinate
            xj = X[:, j]
            r = y - X @ w + xj * w[j]         # residual without j
            w[j] = xj @ r / (xj @ xj)         # Eq. 5.6-2
            if k <= show:
                print(f"  sweep {k}, coordinate {j}: {w.round(4)}")
        if np.abs(w - old).max() < tol:
            return w, k
    return w, None


X = np.c_[np.ones(5), x]
w, k = sweeps(X, show=3)
print(f"raw x:     {k} sweeps, w = {w.round(4)}")
Xc = np.c_[np.ones(5), x - 3]                 # centred feature
w, k = sweeps(Xc, show=1)
print(f"centred x: {k} sweeps, w = {w.round(4)}")
