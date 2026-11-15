# Problem 30B.2: noise through alpha or through WhiteKernel.
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.gaussian_process.kernels import ConstantKernel as C

X = np.array([1., 2, 3, 5, 6, 8])[:, None]
y = np.array([0.2, 1.1, 1.5, 0.4, -0.6, 0.3])
base = C(0.7366, "fixed") * RBF(1.3672, "fixed")
ga = GPR(base, alpha=0.04, optimizer=None).fit(X, y)
gw = GPR(base + WhiteKernel(0.04, "fixed"), alpha=1e-10,
         optimizer=None).fit(X, y)

grid = np.linspace(0, 10, 120)[:, None]
ma, sa = ga.predict(grid, return_std=True)
mw, sw = gw.predict(grid, return_std=True)
print("max |mean difference|:", f"{np.abs(ma - mw).max():.1e}")
print("log marginal likelihood:",
      round(ga.log_marginal_likelihood_value_, 6),
      round(gw.log_marginal_likelihood_value_, 6))
for x0 in (3.0, 4.0):
    i = np.argmin(np.abs(grid[:, 0] - x0))
    print(f"x = {grid[i, 0]:.3f}: std alpha {sa[i]:.4f}, "
          f"std White {sw[i]:.4f}, "
          f"variance difference {sw[i]**2 - sa[i]**2:.6f}")
dev = np.abs(sw**2 - sa**2 - 0.04).max()
print("largest |var_White - var_alpha - 0.04|:", f"{dev:.1e}")
i = np.argmin(np.abs(grid[:, 0] - 3.0))
print("alpha model, std of a NEW reading at x = 3.025:",
      round(float(np.sqrt(sa[i]**2 + 0.04)), 4))
