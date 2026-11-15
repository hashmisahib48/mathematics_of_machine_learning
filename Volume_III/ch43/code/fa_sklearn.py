# Factor analysis: an exact one-factor case, then PCA against FA
import numpy as np
from sklearn.decomposition import FactorAnalysis, PCA

# (1) three features with the correlation matrix of Example 43.8-3
R = np.array([[1, .72, .54], [.72, 1, .48], [.54, .48, 1]])
rng = np.random.RandomState(0)
Z = rng.standard_normal((600, 3)); Z -= Z.mean(axis=0)
Lc = np.linalg.cholesky(Z.T @ Z / 600)
Z = Z @ np.linalg.inv(Lc).T                        # cov = I
X = Z @ np.linalg.cholesky(R).T                    # covariance = R
for tol in (1e-2, 1e-10):
    fa = FactorAnalysis(1, svd_method="lapack", tol=tol,
                        max_iter=5000).fit(X)
    print(f"tol {tol:g}: n_iter {fa.n_iter_:3d}  loadings",
          np.abs(fa.components_[0]).round(4), " psi",
          fa.noise_variance_.round(4))

# (2) six features, one factor, very unequal noise
load = np.array([.9, .8, .7, .6, .5, .2])
psi = np.array([.2, .2, .2, .2, .2, 3.0])
z = rng.standard_normal((2000, 1))
X = z @ load[None] + rng.standard_normal((2000, 6)) * np.sqrt(psi)
fa = FactorAnalysis(1, random_state=0).fit(X)
pca = PCA(1).fit(X)
print("true loadings  ", load)
print("FA loadings    ", np.abs(fa.components_[0]).round(2))
print("PCA direction  ", np.abs(pca.components_[0]).round(2))
print("FA noise var   ", fa.noise_variance_.round(2))
print("PCA noise var  ", round(pca.noise_variance_, 2))
print("mean log-likelihood: FA", round(fa.score(X), 3),
      " PCA", round(pca.score(X), 3))
