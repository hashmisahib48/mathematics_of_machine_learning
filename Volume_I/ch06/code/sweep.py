# Training and test error against polynomial degree.
import numpy as np
from numpy.polynomial import Polynomial

rng = np.random.default_rng(7)

def truth(x):                       # the curve behind the data
    return np.sin(2 * np.pi * x)

def draw(n):                        # n noisy samples, sigma = 0.25
    x = np.sort(rng.uniform(0, 1, n))
    return x, truth(x) + rng.normal(0, 0.25, n)

x_tr, y_tr = draw(15)               # 15 samples for fitting
x_te, y_te = draw(400)              # 400 held-back samples

def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))

print("degree  knobs  train RMSE  test RMSE")
for d in (0, 1, 2, 3, 4, 5, 7, 9, 11, 14):
    f = Polynomial.fit(x_tr, y_tr, d)       # least squares fit
    print(f"{d:6d} {d + 1:6d} {rmse(y_tr, f(x_tr)):11.3f}"
          f" {rmse(y_te, f(x_te)):10.3f}")
