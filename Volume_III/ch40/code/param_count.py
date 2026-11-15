# Free parameters and BIC of a Gaussian mixture, from the formulas.
import math
import numpy as np
from sklearn.mixture import GaussianMixture
from people import people_2d


def n_free(ct, K, p):                       # Eq. 40.6-1
    cov = {"full": K * p * (p + 1) // 2, "tied": p * (p + 1) // 2,
           "diag": K * p, "spherical": K}[ct]
    return (K - 1) + K * p + cov


X, _ = people_2d()
n = len(X)
for ct in ["full", "tied", "diag", "spherical"]:
    gm = GaussianMixture(3, covariance_type=ct, random_state=0)
    gm.fit(X)
    q = n_free(ct, 3, 2)
    assert q == gm._n_parameters()          # private helper: a check
    ll = n * gm.score(X)                    # ln L, Eq. 40.3-3
    bic = -2 * ll + q * math.log(n)         # Eq. 40.6-2
    aic = -2 * ll + 2 * q
    assert np.isclose(bic, gm.bic(X)) and np.isclose(aic, gm.aic(X))
    print(f"{ct:<10s} K=3 p=2: {q:3d} parameters")
for ct in ["full", "tied", "diag", "spherical"]:
    print(f"{ct:<10s} K=3 p=13: {n_free(ct, 3, 13):4d}",
          f"  K=10 p=20: {n_free(ct, 10, 20):5d}")
