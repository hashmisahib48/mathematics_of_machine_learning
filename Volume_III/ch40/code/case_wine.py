import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score

X, y = load_wine(return_X_y=True)       # y is used only to judge
X = StandardScaler().fit_transform(X)   # 178 wines, 13 features
types = ["full", "tied", "diag", "spherical"]
print("BIC       K =", "".join(f"{k:7d}" for k in range(1, 7)))
fits = {}
for ct in types:
    row = []
    for k in range(1, 7):
        gm = GaussianMixture(k, covariance_type=ct, n_init=5,
                             tol=1e-5, max_iter=500, random_state=0)
        fits[ct, k] = gm.fit(X)
        row.append(gm.bic(X))
    print(f"{ct:<14s}", "".join(f"{b:7.0f}" for b in row))

print("\nK = 3: agreement with the three cultivars")
for ct in types:
    gm = fits[ct, 3]
    ari = adjusted_rand_score(y, gm.predict(X))
    print(f"{ct:<10s} BIC {gm.bic(X):5.0f} adjusted Rand {ari:.3f}")
