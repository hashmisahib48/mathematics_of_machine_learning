# Problem 4B.3: ridge regression on a column of ones
import numpy as np
from sklearn.linear_model import Ridge

y = np.array([2.0, 3.0, 3.0, 4.0])
X = np.ones((4, 1))                   # the feature is 1
n = y.size

print("alpha  Ridge  sum(y)/(n+alpha)  n/(n+alpha)  3n/(n+alpha)")
for a in (0, 1, 4, 16):
    m = Ridge(alpha=a, fit_intercept=False).fit(X, y).coef_[0]
    form = y.sum() / (n + a)          # minimiser of Eq. 4.9-5
    w = n / (n + a)                   # weight on the sample mean
    print(f"{a:5d} {m:7.4f} {form:12.4f} {w:14.4f}"
          f" {y.mean() * w:12.4f}")

# the derivative of sum (y - m)^2 + alpha m^2 vanishes at the fit
for a in (1, 4):
    m = Ridge(alpha=a, fit_intercept=False).fit(X, y).coef_[0]
    g = -2 * np.sum(y - m) + 2 * a * m
    print(f"alpha = {a}: gradient at the fit = {g:+.1e}")

r = Ridge(alpha=4).fit(X, y)          # default intercept
print(f"with an intercept: coef = {r.coef_[0]:.4f}, "
      f"intercept = {r.intercept_:.4f}")
