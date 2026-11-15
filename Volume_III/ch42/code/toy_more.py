# The numbers of Examples 42.6-1 and 42.6-2 that Listing 42.1 does not print.
import numpy as np
from sklearn.decomposition import PCA

X = np.array([[27, 4], [29, 6], [30, 6], [31, 8], [33, 6]], float)
mu = X.mean(axis=0)
Xc = X - mu
pca = PCA(2).fit(X)
w1 = pca.components_[0]
Z = pca.transform(X)
rebuilt = mu + np.outer(Z[:, 0], w1)               # one component
resid = X - rebuilt
print("rebuilt (k=1)\n", rebuilt.round(4))
print("squared residual per fish", (resid ** 2).sum(axis=1).round(4),
      " total", (resid ** 2).sum().round(4))
U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
U = U * np.sign(Vt.max(axis=1) + Vt.min(axis=1))   # same signs as PCA
print("U columns\n", U.T.round(4))
print("U orthonormal:", np.allclose(U.T @ U, np.eye(2)))
Zw = PCA(2, whiten=True).fit_transform(X)
print("whitened scores\n", Zw.T.round(4))
print("= sqrt(n-1) * U:", np.allclose(np.abs(Zw), 2 * np.abs(U)))
F = np.array([[32.0, 5.0]])
zF = pca.transform(F)
print("F scores", zF.round(4), " whitened",
      PCA(2, whiten=True).fit(X).transform(F).round(4))
print("rank-1 copy of Xc: squared Frobenius error",
      round(float(((Xc - np.outer(Z[:, 0], w1)) ** 2).sum()), 4),
      " sigma_2^2 =", round(float(s[1] ** 2), 4))
print("PCA(1).get_covariance() =\n",
      PCA(1).fit(X).get_covariance().round(4))
print("S =", np.cov(X.T).tolist())
