"""Data and errors for Figure 1.3-1: polynomials of degree 1, 3 and 9 fitted to 12 noisy points of a smooth curve."""
import numpy as np
rng = np.random.default_rng(0)
f = lambda x: np.sin(1.3 * x) + 0.3 * x
x = np.sort(rng.uniform(0, 4, 12)); y = f(x) + rng.normal(0, 0.25, 12)
xt = np.sort(rng.uniform(x.min(), x.max(), 200)); yt = f(xt) + rng.normal(0, 0.25, 200)
np.savez('overfit.npz', x=x, y=y, xt=xt, yt=yt)
for d in (1, 3, 9):
    c = np.polyfit(x, y, d); e = lambda X, Y: float(np.sqrt(np.mean((np.polyval(c, X) - Y) ** 2)))
    print(f'degree {d}: training RMSE {e(x, y):.3f}, test RMSE {e(xt, yt):.3f}')
