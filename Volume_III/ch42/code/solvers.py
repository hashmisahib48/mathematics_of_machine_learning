# Which svd_solver does 'auto' pick, and how good is 'randomized'?
import numpy as np
from scipy import sparse
from sklearn.datasets import make_low_rank_matrix
from sklearn.decomposition import PCA

rng = np.random.default_rng(0)
cases = [("2000 x 50, k=10", (2000, 50), 10),
         ("300 x 100, k=10", (300, 100), 10),
         ("1000 x 800, k=10", (1000, 800), 10),
         ("1000 x 800, k=0.9", (1000, 800), 0.9),
         ("1000 x 800, k=700", (1000, 800), 700),
         ("600 x 100, k='mle'", (600, 100), "mle")]
for name, shape, k in cases:
    p = PCA(n_components=k).fit(rng.normal(size=shape))
    print(f"{name:20s}-> {p._fit_svd_solver}")
Xs = sparse.random(500, 80, density=0.1, format="csr",
                   random_state=0)
print(f"{'sparse 500 x 80':20s}->",
      PCA(n_components=5).fit(Xs)._fit_svd_solver)

X = make_low_rank_matrix(n_samples=1000, n_features=800,
                         effective_rank=30, tail_strength=0.5,
                         random_state=0)
ref = PCA(10, svd_solver="full").fit(X).explained_variance_
print("iterated_power  max relative error of the 10 variances")
for q in (0, 1, 2, 4, 7):
    v = PCA(10, svd_solver="randomized", iterated_power=q,
            random_state=0).fit(X).explained_variance_
    print(f"{q:14d}  {np.max(np.abs(v - ref) / ref):.2e}")
cov = PCA(10, svd_solver="covariance_eigh").fit(X[:, :50])
full = PCA(10, svd_solver="full").fit(X[:, :50])
d = np.abs(cov.explained_variance_ - full.explained_variance_)
print("covariance_eigh vs full:",
      f"{(d / full.explained_variance_).max():.1e}")
