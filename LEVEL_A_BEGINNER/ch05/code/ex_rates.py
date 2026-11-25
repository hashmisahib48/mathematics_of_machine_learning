# Learning curves: the cost J after each iteration for three rates.
import numpy as np
from sklearn.datasets import load_diabetes
from gd import LinearRegressionGD

X, y = load_diabetes(return_X_y=True)
Z = (X - X.mean(axis=0)) / X.std(axis=0)       # standardized features
for eta in (0.02, 0.2, 0.26):
    m = LinearRegressionGD(eta=eta, n_iter=5000, tol=1e-3)
    m.fit(Z.tolist(), y.tolist())
    c = m.cost_
    print(f"eta {eta:4.2f}: J1 {c[0]:.0f}  J10 {c[9]:.4g}"
          f"  J_end {c[-1]:.6g}  iterations {m.n_iter_}")
