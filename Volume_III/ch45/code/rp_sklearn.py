# Random projection in scikit-learn on the invented spectra.
import numpy as np
from spectra import make_spectra
from sklearn.model_selection import train_test_split
from sklearn.random_projection import (GaussianRandomProjection,
    SparseRandomProjection, johnson_lindenstrauss_min_dim as jl)

X, y = make_spectra()
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)
n, p = Xtr.shape
print("train", Xtr.shape, " JL dims for eps 0.5, 0.3, 0.1:",
      jl(n, eps=[0.5, 0.3, 0.1]))

# one fixed pair of samples, 500 different random matrices
k, d2 = 30, ((Xtr[0] - Xtr[1]) ** 2).sum()
for name, cls in (("Gaussian", GaussianRandomProjection),
                  ("Sparse", SparseRandomProjection)):
    q = []
    for seed in range(500):
        Z = cls(k, random_state=seed).fit_transform(Xtr[:2])
        q.append(((Z[0] - Z[1]) ** 2).sum() / d2)    # Eq. 45.3-2
    q = np.array(q)
    print(f"{name:9s} mean {q.mean():.3f}  sd {q.std():.3f} "
          f"(theory {np.sqrt(2 / k):.3f})")
rp = SparseRandomProjection(k, random_state=0).fit(Xtr)
print("sparse: density_", rp.density_, " values",
      np.unique(rp.components_.data).round(4))

g = GaussianRandomProjection(
    30, random_state=0, compute_inverse_components=True).fit(Xtr)
Z = g.transform(Xte)
back = g.transform(g.inverse_transform(Z))
print("inverse_components_", g.inverse_components_.shape)
print("transform(inverse_transform(Z)) == Z:", np.allclose(back, Z))
