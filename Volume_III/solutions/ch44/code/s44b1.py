# Problem 44B.1: classical MDS on standardised data is PCA
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.manifold import ClassicalMDS
from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(load_wine().data)  # 178 x 13
n = len(X)
pca = PCA(2).fit(X)
Zp = pca.transform(X)                                 # PCA scores
mds = ClassicalMDS(2).fit(X)                          # Euclidean
Zm = mds.embedding_
sign = np.sign((Zp * Zm).sum(axis=0))                 # match signs
print("signs of the two columns:", sign)
print(f"largest |difference|: {np.abs(Zp - sign * Zm).max():.1e}")
print("eigenvalues_      :", mds.eigenvalues_.round(3))
print("explained_variance:", pca.explained_variance_.round(3))
print("ratio:", (mds.eigenvalues_ / pca.explained_variance_).round(6),
      " n - 1 =", n - 1)

# the same eigenvalues from B = -J D^2 J / 2, built by hand
D2 = squareform(pdist(X)) ** 2                        # squared dist.
J = np.eye(n) - np.ones((n, n)) / n                   # centring
B = -0.5 * J @ D2 @ J
lam = np.linalg.eigvalsh(B)[::-1]
print("top eigenvalues of B:", lam[:3].round(3), " rank:",
      int((lam > 1e-8).sum()))
print("sum of squared singular values of X:",
      (np.linalg.svd(X, compute_uv=False)[:2] ** 2).round(3))
print("trace of B:", lam.sum().round(1), "= n p =", n * 13)
print("share of the first two:", (lam[:2].sum() / lam.sum()).round(4),
      " PCA:", pca.explained_variance_ratio_.sum().round(4))
