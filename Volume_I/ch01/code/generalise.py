# Training error against test error for polynomial fits.
import numpy as np
from numpy.polynomial import Polynomial
rng = np.random.default_rng(4)
def truth(x):                   # the smooth curve behind the data
    return np.sin(2 * np.pi * x)
x_tr = np.sort(rng.uniform(0, 1, 12))        # 12 training points
y_tr = truth(x_tr) + rng.normal(0, 0.25, 12)
x_te = np.sort(rng.uniform(0, 1, 40))        # 40 new points
y_te = truth(x_te) + rng.normal(0, 0.25, 40)
def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))
print("degree   train RMSE   test RMSE")
for degree in (0, 1, 3, 9):
    f = Polynomial.fit(x_tr, y_tr, degree)   # learn the knobs
    print(f"{degree:6d} {rmse(y_tr, f(x_tr)):12.3f} "
          f"{rmse(y_te, f(x_te)):11.3f}")
