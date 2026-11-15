# Problem 11B.1: the lasso with distance in km, in m, standardised
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

dist = np.array([1., 2, 3, 4, 5])
lights = np.array([2., 1, 4, 3, 5])
t = np.array([3., 4, 6, 7, 10])
n = 5
sets = {"km": np.c_[dist, lights], "m": np.c_[1000 * dist, lights]}
for k in ("km", "m"):                        # standardised versions
    Z = StandardScaler().fit_transform(sets[k])
    sets[k + ", standardised"] = Z

for name, X in sets.items():
    m = Lasso(alpha=1.0, tol=1e-12, max_iter=100000).fit(X, t)
    rmse = np.sqrt(((t - m.predict(X)) ** 2).mean())
    Xc, yc = X - X.mean(0), t - t.mean()
    v = (Xc ** 2).mean(0)                  # v_j, Eq. 11.3-3
    c = Xc.T @ yc / n                      # c_j, Eq. 11.3-3
    w1 = np.sign(c[0]) * max(abs(c[0]) - 1, 0) / v[0]  # 11.3-6
    g = Xc[:, 0] @ Xc[:, 1] / n            # mean product x1 x2
    z2 = c[1] - g * w1                     # Eq. 11.3-10
    print(name)
    print(f"  w = {np.round(m.coef_, 4)}  b = {m.intercept_:.4f}"
          f"  rmse = {rmse:.3f}")
    print(f"  v1 = {v[0]:.4g}  c1 = {c[0]:.4g}  "
          f"w1 (if w2 = 0) = {w1:.4f}  z2 = {z2:.4f}")
