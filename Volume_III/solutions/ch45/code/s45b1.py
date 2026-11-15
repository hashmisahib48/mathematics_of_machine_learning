# Problem 45B.1: random projections of the digits, 5-NN accuracy
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.random_projection import (GaussianRandomProjection,
                                       SparseRandomProjection)

X, y = load_digits(return_X_y=True)          # unscaled, p = 64
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                      random_state=0, stratify=y)
def knn(A, B):                               # test accuracy of 5-NN
    return KNeighborsClassifier(5).fit(A, ytr).score(B, yte)

print("raw 64 pixels: %.3f" % knn(Xtr, Xte))
print("  k   Gaussian        sparse          PCA    sqrt(2/k)")
for k in (4, 8, 16, 32):
    cells = []
    for cls in (GaussianRandomProjection, SparseRandomProjection):
        acc = []
        for seed in range(10):             # ten matrices R
            rp = cls(k, random_state=seed).fit(Xtr)   # shape only
            acc.append(knn(rp.transform(Xtr), rp.transform(Xte)))
        cells.append("%.3f +- %.3f" % (np.mean(acc), np.std(acc)))
    pca = PCA(k, random_state=0).fit(Xtr)    # the data-driven rival
    a_pca = knn(pca.transform(Xtr), pca.transform(Xte))
    print(f"{k:3d}   {cells[0]}   {cells[1]}   {a_pca:.3f}"
          f"   {np.sqrt(2/k):.3f}")
