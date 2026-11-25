# Vectors and matrices with NumPy.
import numpy as np

u = np.array([3.0, 4.0]); v = np.array([4.0, 3.0])
print("u . v =", u @ v, "  |u| =", np.linalg.norm(u))
cos = u @ v / (np.linalg.norm(u) * np.linalg.norm(v))
print("angle:", round(np.degrees(np.arccos(cos)), 2), "degrees")
X = np.array([[1.0, 2.0], [3.0, 1.0], [2.0, 2.0]])     # 3 x 2
w = np.array([0.5, 2.0])
print("X w =", X @ w)                    # one dot product per row
A = X.T @ X                              # 2 x 2, symmetric
print("X^T X =", A.tolist())
c = np.array([4.0, 5.0])
print("solution of A z = c:", np.linalg.solve(A, c).round(4))
