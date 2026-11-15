# Problem 14B.3: TweedieRegressor reproduces three other estimators
import numpy as np
from sklearn.linear_model import (GammaRegressor, PoissonRegressor,
                                  Ridge, TweedieRegressor)

X = np.arange(6.0)[:, None]
y = np.array([2.1, 2.9, 4.6, 6.2, 8.8, 14.5])
n, a = len(y), 0.2
kw = dict(alpha=a, tol=1e-12, max_iter=1000)

pairs = {
    "power=1 / PoissonRegressor": (TweedieRegressor(power=1, **kw),
                                   PoissonRegressor(**kw)),
    "power=2 / GammaRegressor": (TweedieRegressor(power=2, **kw),
                                 GammaRegressor(**kw)),
    "power=0 / Ridge(n*alpha)": (TweedieRegressor(power=0, **kw),
                                 Ridge(alpha=n * a)),
}
print("pair                        slope A  slope B  d(slope) d(b)")
for name, (A, B) in pairs.items():
    A.fit(X, y)
    B.fit(X, y)
    ds = abs(A.coef_[0] - B.coef_[0])
    db = abs(A.intercept_ - B.intercept_)
    print(f"{name:27s} {A.coef_[0]:7.4f} {B.coef_[0]:8.4f}"
          f" {ds:8.1e} {db:8.1e}")
