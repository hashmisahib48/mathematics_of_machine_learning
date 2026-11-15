# The crack-count problem with PoissonRegressor.
import numpy as np
from sklearn.linear_model import PoissonRegressor, Ridge
from sklearn.linear_model import TweedieRegressor
from sklearn.metrics import mean_poisson_deviance, d2_tweedie_score

X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])   # shape (5, 1)
y = np.array([1.0, 2.0, 2.0, 4.0, 6.0])

model = PoissonRegressor(alpha=0.0, tol=1e-8).fit(X, y)
mu = model.predict(X)
print("coef_      =", np.round(model.coef_, 4))
print(f"intercept_ = {model.intercept_:.4f}")
print("n_iter_    =", model.n_iter_)
print("exp(coef_) =", np.round(np.exp(model.coef_), 4))
print("score (D2) =", round(model.score(X, y), 4))
print("d2_tweedie =", round(d2_tweedie_score(y, mu, power=1), 4))
print("mean dev.  =", round(mean_poisson_deviance(y, mu), 4))

newton = PoissonRegressor(alpha=0.0, tol=1e-8,
                          solver="newton-cholesky").fit(X, y)
print(f"newton-cholesky: {newton.coef_[0]:.4f} "
      f"in {newton.n_iter_} steps")

n, a = len(y), 0.3                        # Tweedie power 0 = ridge
tw = TweedieRegressor(power=0, alpha=a).fit(X, y)
rd = Ridge(alpha=n * a).fit(X, y)
print("Tweedie(0), Ridge(n*alpha):", np.round(tw.coef_, 4),
      np.round(rd.coef_, 4))
