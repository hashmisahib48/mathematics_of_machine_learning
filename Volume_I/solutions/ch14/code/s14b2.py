# Problem 14B.2: the unit of y in GammaRegressor and PoissonRegressor
import numpy as np
from sklearn.linear_model import GammaRegressor, PoissonRegressor
from sklearn.metrics import mean_tweedie_deviance

X = np.arange(6.0)[:, None]
y = np.array([2.1, 2.9, 4.6, 6.2, 8.8, 14.5])
c = 1000.0

print("model    alpha  slope(y)  slope(1000y)  b(1000y)-b(y)")
models = {"gamma": GammaRegressor, "Poisson": PoissonRegressor}
for name, Model in models.items():
    for a in (0.0, 1.0):
        m1 = Model(alpha=a, tol=1e-12, max_iter=1000).fit(X, y)
        m2 = Model(alpha=a, tol=1e-12, max_iter=1000).fit(X, c * y)
        db = m2.intercept_ - m1.intercept_
        print(f"{name:8s} {a:5.1f} {m1.coef_[0]:9.4f}"
              f" {m2.coef_[0]:12.4f} {db:14.4f}")
print(f"ln(1000) = {np.log(c):.4f}")

# the scaling law d(cy, cmu) = c^(2-p) d(y, mu)
mu = np.full(6, 5.0)
for p in (1, 2):
    d1 = mean_tweedie_deviance(y, mu, power=p)
    d2 = mean_tweedie_deviance(c * y, c * mu, power=p)
    print(f"power {p}: deviance {d1:.4f} -> {d2:.4f}, "
          f"ratio {d2 / d1:.1f}")
