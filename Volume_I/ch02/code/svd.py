# Singular value decomposition, pseudo-inverse, conditioning (2.10).
import numpy as np

A = np.array([[2.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
U, s, Vt = np.linalg.svd(A, full_matrices=False)    # Eq. 2.10-1
print("singular values:", np.round(s, 4))
lam = np.linalg.eigvalsh(A.T @ A)[::-1]
print("sqrt of eigenvalues of A^T A:", np.round(np.sqrt(lam), 4))
print("U =\n", np.round(U, 4))
print("Vt =\n", np.round(Vt, 4))
print("A = U diag(s) V^T:", np.allclose(U * s @ Vt, A))
print("cond(A) =", round(float(s[0] / s[-1]), 4),
      " numpy:", round(float(np.linalg.cond(A)), 4))    # Eq. 2.10-7

A1 = s[0] * np.outer(U[:, 0], Vt[0])       # best rank 1, Eq. 2.10-4
print("rank-1 error:", round(float(np.linalg.norm(A - A1)), 4),
      "  second singular value:", round(float(s[1]), 4))

pinv = Vt.T @ np.diag(1 / s) @ U.T              # Eq. 2.10-6
print("pinv matches:", np.allclose(pinv, np.linalg.pinv(A)))
print("pinv @ b:", np.round(pinv @ np.array([3.0, 3.0, 3.0]), 4))

rng = np.random.default_rng(0)       # two features, correlation rho
x1 = rng.normal(size=50)
for rho in (0.0, 0.9, 0.999):
    x2 = rho * x1 + np.sqrt(1 - rho**2) * rng.normal(size=50)
    X = np.column_stack([x1, x2])
    c1, c2 = np.linalg.cond(X), np.linalg.cond(X.T @ X)
    print(f"rho = {rho:5.3f}   cond(X) = {c1:6.1f}"
          f"   cond(X^T X) = {c2:8.1f}")
