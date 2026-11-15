# Kernel PCA of four points on a line, kernel k(x, z) = (x z)^2.
import numpy as np
from sklearn.decomposition import KernelPCA

x = np.array([[-2.0], [-1.0], [1.0], [2.0]])
n = len(x)
K = (x @ x.T) ** 2                          # Gram matrix
J = np.eye(n) - np.ones((n, n)) / n         # centring matrix
Kc = J @ K @ J                              # Eq. 42.11-2
print("K row 1  ", K[0], "\nKc row 1 ", Kc[0])
lam, A = np.linalg.eigh(Kc)
lam, A = lam[::-1], A[:, ::-1]              # largest first
print("eigenvalues of Kc", lam.round(4) + 0.0)
s = np.sqrt(lam[0]) * A[:, 0]               # scores of the 4 points
print("scores sqrt(lam) * a:", s.round(4))

kp = KernelPCA(n_components=1, kernel="poly", degree=2, gamma=1,
               coef0=0).fit(x)
t = kp.transform(x)[:, 0]
print("KernelPCA scores    :", t.round(4), kp.eigenvalues_.round(4))
z = np.array([[0.0]])                       # a new point
kz = (z @ x.T) ** 2                         # its kernel row
kzc = kz - kz.mean() - K.mean(axis=0) + K.mean()    # Eq. 42.11-3
new = (kzc @ A[:, :1] / np.sqrt(lam[0]))[0, 0]
print("new point by hand", round(new, 4),
      " KernelPCA", kp.transform(z).round(4)[0, 0])
print("same up to sign:", np.allclose(abs(s), abs(t)),
      np.isclose(abs(new), abs(kp.transform(z)[0, 0])))
