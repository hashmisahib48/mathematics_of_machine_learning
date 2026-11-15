import numpy as np
from sklearn.mixture import GaussianMixture
from people import people_2d

X, _ = people_2d()
Ks = range(1, 9)
print(f"{'BIC, K =':<10s}", "".join(f"{k:6d}" for k in Ks))
best = (np.inf, None, None)
for ct in ["full", "tied", "diag", "spherical"]:
    bic = []
    for k in Ks:
        gm = GaussianMixture(k, covariance_type=ct, n_init=3,
                             tol=1e-5, max_iter=500,
                             random_state=0).fit(X)
        bic.append(gm.bic(X))
    print(f"{ct:<10s}", "".join(f"{b:6.0f}" for b in bic))
    if min(bic) < best[0]:
        best = (min(bic), ct, 1 + int(np.argmin(bic)))
print(f"lowest BIC {best[0]:.0f}: {best[1]} covariances,",
      f"{best[2]} components")
