# Eigenvalues and eigenvectors of a symmetric matrix (Section 2.9).
import numpy as np

S = np.array([[5.0, 2.0], [2.0, 2.0]])
lam, V = np.linalg.eigh(S)                        # ascending order
print("eigenvalues:", lam)                       # Eq. 2.9-3
print("eigenvectors (columns):\n", np.round(V, 4))
print("S v = lambda v:", np.allclose(S @ V, V * lam))
print("V^T V = I:", np.allclose(V.T @ V, np.eye(2)))
print("trace =", np.trace(S), "  sum of eigenvalues =", lam.sum())
print("det   =", round(float(np.linalg.det(S)), 6),
      "  product of eigenvalues =", round(float(lam.prod()), 6))

S2 = sum(l * np.outer(v, v) for l, v in zip(lam, V.T))  # Eq. 2.9-5
print("sum of lambda q q^T equals S:", np.allclose(S2, S))

rng = np.random.default_rng(1)                  # Eq. 2.9-6
Z = rng.multivariate_normal([0.0, 0.0], S, size=2000)
C = np.cov(Z.T)
print("sample covariance:", np.round(C, 2).tolist())
print("its eigenvalues:", np.round(np.linalg.eigvalsh(C), 3))
