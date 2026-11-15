# Bias, variance and noise by refitting on many training sets.
import numpy as np
from numpy.polynomial import Polynomial

rng = np.random.default_rng(60)
sigma, n, n_sets = 0.25, 20, 500
x_tr = (np.arange(n) + 0.5) / n            # fixed, evenly spaced x
x0 = np.linspace(0, 1, 101)                # where fits are judged
f0 = np.sin(2 * np.pi * x0)                # the true curve there

def fit_all(d):                            # one row per set
    P = np.empty((n_sets, x0.size))
    for s in range(n_sets):
        y = np.sin(2 * np.pi * x_tr) + rng.normal(0, sigma, n)
        P[s] = Polynomial.fit(x_tr, y, d)(x0)   # fresh noise
    return P

print("degree  bias^2  variance  noise   sum   simulated MSE")
for d in (0, 1, 3, 5, 9):
    P = fit_all(d)
    bias2 = np.mean((P.mean(axis=0) - f0) ** 2)    # Eq. 6.3-4
    var = np.mean(P.var(axis=0))                   # Eq. 6.3-4
    y_new = f0 + rng.normal(0, sigma, P.shape)     # fresh targets
    mse = np.mean((y_new - P) ** 2)                # direct estimate
    print(f"{d:6d} {bias2:7.4f} {var:9.4f} {sigma**2:6.4f}"
          f" {bias2 + var + sigma**2:6.4f} {mse:12.4f}")
