# Problem 12B.2: an exact copy of a feature, BayesianRidge and ARD
import numpy as np
from sklearn.linear_model import ARDRegression, BayesianRidge

rng = np.random.default_rng(1)
n = 50
u = rng.normal(size=n)                         # the useful feature
X = np.c_[u, u, rng.normal(size=(n, 3))]       # u, copy of u, 3 noise
y = 2 * u + rng.normal(0, 0.5, n)

br = BayesianRidge().fit(X, y)
ard = ARDRegression().fit(X, y)
names = ["u", "copy", "z1", "z2", "z3"]
print("column            ", *[f"{c:>7s}" for c in names])
print("BayesianRidge coef", *[f"{c:7.3f}" for c in br.coef_])
print("ARDRegression coef", *[f"{c:7.3f}" for c in ard.coef_ + 0.0])
print("ARD lambda_       ", *[f"{c:7.3g}" for c in ard.lambda_])
print("BayesianRidge: lambda_ = %.3f, alpha_ = %.3f (noise sd %.3f)"
      % (br.lambda_, br.alpha_, br.alpha_ ** -0.5))
print("sum of the two weights: BR %.3f, ARD %.3f"
      % (br.coef_[:2].sum(), ard.coef_[:2].sum()))
w = np.linalg.lstsq(X[:, :2], y, rcond=None)[0]  # min-norm
print("least squares on [u, copy]:", np.round(w, 3))
print("least squares on u alone:", round(float(u @ y / (u @ u)), 3))
print("ARD columns with coef exactly 0:",
      np.flatnonzero(ard.coef_ == 0))
print("sd of a least-squares weight on a noise column: 0.5/sqrt(n) =",
      round(0.5 / n ** 0.5, 4))
