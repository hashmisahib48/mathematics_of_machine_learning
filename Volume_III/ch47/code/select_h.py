# Kernel and bandwidth for 200 invented waiting times.
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KernelDensity


def waits(n, seed):            # short waits (35 %) and long ones
    rng = np.random.default_rng(seed)
    short = rng.random(n) < 0.35
    t = np.where(short, rng.normal(54, 5, n), rng.normal(80, 6, n))
    return t[:, None]


X, X_test = waits(200, 1), waits(5000, 2)
cv = KFold(10, shuffle=True, random_state=0)
start = {"gaussian": 1.0, "epanechnikov": 3.0}  # finite scores
for kernel, lo in start.items():
    grid = {"bandwidth": np.arange(lo, 10.01, 0.25)}
    est = KernelDensity(kernel=kernel)
    gs = GridSearchCV(est, grid, cv=cv).fit(X)
    best = gs.best_estimator_
    print(f"{kernel:12s} h = {best.bandwidth:.2f}  "
          f"cv = {gs.best_score_ / 20:.4f}  "
          f"test = {best.score(X_test) / len(X_test):.4f}")

sd = X.std()                      # the rules return only a factor
for rule in ["scott", "silverman"]:
    k = KernelDensity(bandwidth=rule).fit(X)
    h = k.bandwidth_ * sd         # Eq. 47.4-4
    k = KernelDensity(bandwidth=h).fit(X)
    print(f"{rule:12s} h = {h:.2f}  test = "
          f"{k.score(X_test) / len(X_test):.4f}")

gmm = GaussianMixture(2, random_state=0).fit(X)
print(f"{'GMM, K = 2':12s} test = {gmm.score(X_test):.4f}")
