# Maximum likelihood for a coin and for a normal distribution.
import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar

k, n = 7, 10                                 # 7 heads in 10 tosses


def neg_loglik(t):
    return -(k * np.log(t) + (n - k) * np.log(1 - t))   # Eq. 4.8-2


grid = np.linspace(0.01, 0.99, 981)
print(f"grid maximum   : {grid[np.argmin(neg_loglik(grid))]:.2f}")
res = minimize_scalar(neg_loglik, bounds=(0.01, 0.99),
                      method="bounded")
print(f"optimiser      : {res.x:.4f}   formula k/n = {k/n:.4f}")
for t in (0.5, 0.6, 0.7, 0.8):
    print(f"L({t}) = {t**k * (1-t)**(n-k):.6f}")

d = np.array([2.0, 4, 4, 6])                 # normal sample
mu, sig = stats.norm.fit(d)                  # maximum likelihood
print(f"norm.fit       : mu = {mu:.4f}, sigma^2 = {sig**2:.4f}")
print(f"formulas       : mu = {d.mean():.4f}, "
      f"sigma^2 = {np.mean((d - d.mean())**2):.4f}")   # Eq. 4.8-6
