# Test of MyGLM against the three scikit-learn regressors.
import numpy as np
from sklearn.linear_model import GammaRegressor, PoissonRegressor
from sklearn.linear_model import TweedieRegressor
from glm_scratch import MyGLM

rng = np.random.default_rng(0)
X = rng.normal(size=(300, 3))
mean = np.exp(0.3 + X @ np.array([0.5, -0.4, 0.2]))
s = rng.uniform(0.2, 2.0, size=300)           # sample weights
count = rng.poisson(mean)
cases = [(1.0, PoissonRegressor(), count),
         (2.0, GammaRegressor(), rng.gamma(2.0, mean / 2.0)),
         (1.5, TweedieRegressor(power=1.5),
          count * rng.gamma(2.0, 0.5, size=300))]
for power, ref, y in cases:
    for a in (0.0, 0.1):
        mine = MyGLM(power, alpha=a).fit(X, y, s)
        ref.set_params(alpha=a, tol=1e-12, max_iter=1000)
        ref.fit(X, y, sample_weight=s)
        d = np.abs(mine.coef_ - ref.coef_).max()
        assert d < 1e-6
        assert abs(mine.intercept_ - ref.intercept_) < 1e-6
        print(f"power {power}  alpha {a:<3}  Newton {mine.n_iter_}"
              f"  L-BFGS {ref.n_iter_:2d}  max|dw| = {d:.1e}")
