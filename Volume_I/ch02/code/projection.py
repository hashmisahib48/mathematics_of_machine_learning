# Projection onto a line, onto a column space, and Chapter 9's data.
import numpy as np

u = np.array([3.0, 1.0])
b = np.array([1.0, 2.0])
c = (u @ b) / (u @ u)                           # Eq. 2.4-2
p = c * u
r = b - p
print(f"line:  c = {c}  p = {p}  r = {r}  u.r = {u @ r}")

A = np.array([[2.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
b = np.array([3.0, 3.0, 3.0])
c = np.linalg.solve(A.T @ A, A.T @ b)           # Eq. 2.4-3
p = A @ c
r = b - p
print("A^T A =", (A.T @ A).tolist(), "  A^T b =", A.T @ b)
print("plane: c =", c, " p =", p, " r =", r)
print("A^T r =", A.T @ r)                       # Eq. 2.4-4
P = A @ np.linalg.inv(A.T @ A) @ A.T            # Eq. 2.4-5
print("P b = p:", np.allclose(P @ b, p),
      "  P P = P:", np.allclose(P @ P, P))

x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])         # delivery data
y = np.array([3.0, 4.0, 6.0, 7.0, 10.0])
X = np.column_stack([np.ones(5), x])            # 5 x 2, Eq. 9.4-2
w, *_ = np.linalg.lstsq(X, y, rcond=None)
r = y - X @ w
print("weights (b, w):", np.round(w, 4))
print("y_hat:", np.round(X @ w, 2))
print("residual:", np.round(r, 2))
print("X^T r:", np.round(X.T @ r, 10) + 0.0)
