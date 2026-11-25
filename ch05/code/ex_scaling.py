# Feature scaling and the speed of gradient descent.
import numpy as np
from gd import LinearRegressionGD

X = np.array([[2, 1], [4, 2], [5, 1], [7, 3], [9, 2]], dtype=float)
y = [14, 23, 24, 35, 38]
Z = (X - X.mean(axis=0)) / X.std(axis=0)       # standardized
for name, D in [("raw", X), ("standardized", Z)]:
    Dc = D - D.mean(axis=0)
    A = Dc.T @ Dc
    eta = 0.9 * len(y) / np.trace(A)           # safe rate, Eq. 5.2-8
    m = LinearRegressionGD(eta=eta, n_iter=10000).fit(D.tolist(), y)
    print(f"{name:12s} eta {eta:.3f}  iterations {m.n_iter_:4d}"
          f"  J = {m.cost_[-1]:.4f}")
