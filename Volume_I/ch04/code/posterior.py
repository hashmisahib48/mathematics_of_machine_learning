# Priors, posteriors and the link to ridge regression.
import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge

# Coin: 7 heads in 10, Beta(2, 2) prior.      Eq. 4.9-3
a, b, k, n = 2, 2, 7, 10
post = stats.beta(a + k, b + n - k)
map_ = (a + k - 1) / (a + b + n - 2)         # mode of the Beta
print(f"posterior Beta({a+k}, {b+n-k}): mean {post.mean():.4f}, "
      f"MAP {map_:.4f}")

# Normal mean: data variance 4, prior N(0, 1), data below.
y = np.array([2.0, 3, 3, 4])
s2, t2, m0 = 4.0, 1.0, 0.0
prec = len(y) / s2 + 1 / t2                  # posterior precision
mean = (y.sum() / s2 + m0 / t2) / prec       # Eq. 4.9-4
print(f"posterior mean {mean:.4f}, variance {1/prec:.4f}")

# Same number from ridge with alpha = s2 / t2.   Eq. 4.9-5
X = np.ones((len(y), 1))
rd = Ridge(alpha=s2 / t2, fit_intercept=False).fit(X, y)
print(f"Ridge(alpha={s2/t2:.0f}) coef_ = {rd.coef_[0]:.4f}")

# Shrunk mean c * x_bar: bias^2, variance, MSE (n=4, mu=1, sd=2).
mu, sd, n = 1.0, 2.0, 4
for c in (1.0, 0.8, 0.5):
    bias, var = (c - 1) * mu, c**2 * sd**2 / n   # Eq. 4.10-1
    print(f"c = {c}: bias^2 {bias**2:.3f}  var {var:.3f}  "
          f"MSE {bias**2 + var:.3f}")
rng = np.random.default_rng(0)
xb = rng.normal(mu, sd, size=(200_000, n)).mean(axis=1)
sim = [np.mean((c * xb - mu) ** 2) for c in (1.0, 0.8, 0.5)]
print("simulated MSE:", np.round(sim, 3))
