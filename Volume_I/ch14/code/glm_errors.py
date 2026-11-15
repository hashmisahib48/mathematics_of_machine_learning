# Common mistakes with the generalised linear models.
import numpy as np
from sklearn.linear_model import GammaRegressor, PoissonRegressor

X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
y = np.array([1.0, 2.0, 2.0, 4.0, 6.0])         # the crack counts

try:
    GammaRegressor().fit(X, np.where(y == 2, 0.0, y))   # zeros
except ValueError as e:
    print("ValueError:", e)

big = 1e5 * X                                    # cycles, not 10^5
for solver in ("lbfgs", "newton-cholesky"):
    m = PoissonRegressor(alpha=0.0, solver=solver, tol=1e-8)
    m.fit(big, y)
    print(f"{solver:16s} n_iter_ = {m.n_iter_:2d}   "
          f"coef_ x 1e5 = {m.coef_[0] * 1e5:.4f}")

for a in (1.0, 0.0):                        # 1.0 is the default
    m = PoissonRegressor(alpha=a, tol=1e-8).fit(X, y)
    print(f"alpha={a}: coef_ = {m.coef_[0]:.4f}, "
          f"D2 = {m.score(X, y):.4f}")
