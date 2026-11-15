# Problem 18B.1: is the projection of MyLDA a whitening? (wine data)
import sys
import numpy as np
from sklearn.datasets import load_wine
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis as LDA)
sys.path.insert(0, "../../../ch18/code")      # Listing 18.1
from lda_scratch import MyLDA

X, y = load_wine(return_X_y=True)
mine, ref = MyLDA().fit(X, y), LDA().fit(X, y)
Zm, Zr = mine.transform(X), ref.transform(X)[:, :2]
pri = np.bincount(y) / len(y)                 # class proportions
for name, Z in (("MyLDA", Zm), ("scikit-learn", Zr)):
    # centre each class, pool with divisor n_k and weights n_k/n
    W = [Z[y == k] - Z[y == k].mean(0) for k in range(3)]
    S = sum(q * np.cov(w.T, bias=True) for q, w in zip(pri, W))
    cen = np.array([Z[y == k].mean(0) for k in range(3)])
    print(name)
    print("  within-class covariance:\n", S.round(4) + 0.0)
    print("  class centroids:\n", cen.round(3))
dz = np.abs(np.abs(Zm) - np.abs(Zr)).max()
print(f"max | |Z_mine| - |Z_sklearn| | = {dz:.1e}")
same = [bool((Zm[:, j] * Zr[:, j]).min() > 0) for j in (0, 1)]
print("signs of the two columns agree?", same)
cen = np.array([Zm[y == k].mean(0) for k in range(3)])
var = pri @ cen ** 2                  # spread of the centroids, S_b
print("ratios (MyLDA, sklearn):", mine.explained_variance_ratio_.round(4),
      ref.explained_variance_ratio_.round(4))
print("share of the centroid spread:", (var / var.sum()).round(4))
d = np.linalg.norm(cen[:, None] - cen[None], axis=2)
print("distances between centroids:\n", d.round(3))
