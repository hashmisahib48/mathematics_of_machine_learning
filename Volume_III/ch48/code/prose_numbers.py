# Numbers quoted in the prose of Chapter 48.
import numpy as np
from scipy.optimize import minimize
from scipy.special import digamma
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.linear_model import SGDOneClassSVM
from sensors import make_sensors

H = lambda n: digamma(n + 1) + np.euler_gamma
def c_lib(n):
    return 0 if n <= 1 else 1 if n == 2 else \
        2 * (np.log(n - 1) + np.euler_gamma) - 2 * (n - 1) / n
print("n   exact 2(H_n-1)   library c(n)")
for n in (2, 3, 4, 8, 16, 64, 256):
    print(f"{n:3d} {2 * (H(n) - 1):12.4f} {c_lib(n):12.4f}")
print("c(256) =", round(c_lib(256), 4), " max_depth(256) =",
      int(np.ceil(np.log2(256))), " c(8) =", round(c_lib(8), 4))
for h in (4.0, c_lib(256), 8.0):
    print(f"psi=256, E h={h:.3f}: s={2 ** (-h / c_lib(256)):.4f}")

X, y = make_sensors()
sc = StandardScaler().fit(X)
Z = sc.transform(X)
ee = EllipticEnvelope(contamination=0.05, random_state=0).fit(Z)
print("EllipticEnvelope location (original units):",
      sc.inverse_transform(ee.location_[None]).round(1),
      "support share", round(ee.support_.mean(), 3))
f = IsolationForest(random_state=0).fit(Z)
print("IsolationForest max_samples_", f.max_samples_,
      "max_depth", int(np.ceil(np.log2(f.max_samples_))))
lo = LocalOutlierFactor(novelty=True).fit(Z)
nov = lo.score_samples(Z)
print("LOF novelty=True on its own training rows: largest abs gap to",
      "negative_outlier_factor_", np.abs(nov - lo.negative_outlier_factor_).max().round(3))

# SGDOneClassSVM minimises J of Eq. 48.4-9 (linear, 60 points)
rng = np.random.RandomState(0)
A = rng.normal(size=(60, 2)) + [3, 2]
nu, n = 0.2, 60
J = lambda v: nu / 2 * v[:2] @ v[:2] + np.maximum(0, v[2] - A @ v[:2]).mean() - nu * v[2]
best = minimize(J, [1, 1, 1.], method="Nelder-Mead",
                options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20000})
print("min J =", round(best.fun, 4), "at w, rho =", best.x.round(3))
for lr, kw in (("constant", {"eta0": 0.01}), ("optimal", {})):
    s = SGDOneClassSVM(nu=nu, tol=None, max_iter=20000, random_state=0,
                       learning_rate=lr, **kw).fit(A)
    print(f"{lr:8s}: J = {J(np.r_[s.coef_, s.offset_]):.4f}",
          "w, rho =", np.r_[s.coef_, s.offset_].round(3),
          "coef_ shape", s.coef_.shape)
mode = X[:, 0] > 65                      # load mode vs idle mode rows
print("EllipticEnvelope median d^2: idle", np.median(ee.dist_[~mode]).round(1),
      "load", np.median(ee.dist_[mode]).round(1))
