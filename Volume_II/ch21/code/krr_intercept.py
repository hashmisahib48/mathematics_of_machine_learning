# KernelRidge has no intercept: two remedies for a level far from 0.
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import rbf_kernel

rng = np.random.default_rng(2)
x = np.sort(rng.uniform(0, 6, 30))[:, None]
f = lambda t: 50 + np.sin(t)               # the truth: level 50
y = f(x).ravel() + 0.2 * rng.normal(size=30)
grid = np.linspace(-1, 7, 200)[:, None]    # reaches beyond the data
inside = (grid.ravel() >= 0) & (grid.ravel() <= 6)


def report(name, pred):
    e = (pred - f(grid).ravel()) ** 2
    print(f"{name:21s} RMSE inside {np.sqrt(e[inside].mean()):6.3f}"
          f"   outside {np.sqrt(e[~inside].mean()):6.3f}")


kw = dict(alpha=0.1, kernel="rbf", gamma=1.0)
report("raw target", KernelRidge(**kw).fit(x, y).predict(grid))

m = y.mean()                               # remedy 1: centre y
fit = KernelRidge(**kw).fit(x, y - m)
report("centred target", fit.predict(grid) + m)

c = 2500.0                                 # remedy 2: k + c
K = rbf_kernel(x, gamma=1.0) + c
mod = KernelRidge(alpha=0.1, kernel="precomputed").fit(K, y)
K_new = rbf_kernel(grid, x, gamma=1.0) + c
report("kernel plus constant", mod.predict(K_new))
