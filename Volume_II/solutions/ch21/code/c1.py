# Problem 21C.1: primal or dual for n = 50 000, p = 20 (counts, and two measurements)
import time, numpy as np
from math import comb
from scipy.linalg import cho_factor, cho_solve
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

n, p = 50_000, 20; m = comb(p + 3, 3)
print("columns of the degree-3 polynomial in %d features: C(%d, 3) = %d" % (p, p + 3, m))
print("DUAL (KernelRidge, RBF)   K: n x n = %.2e entries = %.1f GB" % (n * n, n * n * 8 / 1e9))
print("   building K  n^2 p = %.2e   Cholesky n^3/3 = %.2e   one prediction: n p = %.1e" % (n * n * p, n ** 3 / 3, n * p))
print("PRIMAL (Ridge on %d columns)   Phi: n x m = %.2e entries = %.2f GB" % (m, n * m, n * m * 8 / 1e9))
print("   Phi^T Phi  n m^2 = %.2e   solve m^3/3 = %.2e   Phi^T Phi stored: %.0f MB   one prediction: m = %d" % (n * m * m, m ** 3 / 3, m * m * 8 / 1e6, m))
print("ratio of operations dual/primal: %.0f ; ratio of memory (K against Phi): %.0f" % ((n ** 3 / 3 + n * n * p) / (n * m * m + m ** 3 / 3), n * n / (n * m)))
rng = np.random.default_rng(0)
nb = 3000; Kb = rng.normal(size=(nb, 40)); Kb = Kb @ Kb.T + np.eye(nb)
t0 = time.perf_counter(); cho_solve(cho_factor(Kb), rng.normal(size=nb)); tb = time.perf_counter() - t0
print("measured: one Cholesky solve with n = %d takes %.2f s; scaled by (n/%d)^3 = %.0f to n = 50 000: about %.0f s (machine dependent)"
      % (nb, tb, nb, (n / nb) ** 3, tb * (n / nb) ** 3))
X = rng.normal(size=(n, p)); y = X[:, 0] * X[:, 1] + X[:, 2] ** 3 + rng.normal(size=n)
t0 = time.perf_counter(); Phi = PolynomialFeatures(3).fit_transform(X); t1 = time.perf_counter()
r = Ridge(alpha=1.0).fit(Phi, y); t2 = time.perf_counter()
print("measured: PolynomialFeatures(3) on 50 000 rows: %d columns, %.1f s; Ridge fit on them: %.1f s (machine dependent)" % (Phi.shape[1], t1 - t0, t2 - t1))
print("training R^2 of the polynomial ridge on this made-up target: %.3f" % r.score(Phi, y))
