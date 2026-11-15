# First- and second-order Taylor expansions of softplus at 0.
import numpy as np

f = lambda x: np.log1p(np.exp(x))          # softplus
f0, f1, f2 = np.log(2), 0.5, 0.25          # f(0), f'(0), f''(0)
t1 = lambda x: f0 + f1 * x                 # Eq. 3.9-1
t2 = lambda x: f0 + f1 * x + 0.5 * f2 * x ** 2   # Eq. 3.9-2
print("   x    exact   1st order  error   2nd order  error")
for x in [0.1, 0.5, 1.0, 2.0]:
    print(f"{x:5.1f} {f(x):8.4f} {t1(x):8.4f} {f(x)-t1(x):8.4f}"
          f" {t2(x):9.4f} {f(x)-t2(x):8.4f}")
