# Rank of a matrix and of the delivery matrix (Section 2.6).
import numpy as np

A = np.array([[2.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
A3 = np.column_stack([A, A[:, 0] + A[:, 1]])      # a3 = a1 + a2
print("rank A  =", np.linalg.matrix_rank(A), " shape", A.shape)
print("rank A3 =", np.linalg.matrix_rank(A3), " shape", A3.shape)

x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
X = np.column_stack([np.ones(5), x])
Xm = np.column_stack([X, 0.621371 * x])          # distance in miles
print("rank X  =", np.linalg.matrix_rank(X),
      "  with miles:", np.linalg.matrix_rank(Xm))
s = np.linalg.svd(Xm, compute_uv=False)
print("singular values:", [f"{z:.3e}" for z in s])
print("rank of X^T X (3 x 3):", np.linalg.matrix_rank(Xm.T @ Xm))

rng = np.random.default_rng(0)             # fewer rows than columns
W = rng.normal(size=(6, 8))
print("6 x 8: rank =", np.linalg.matrix_rank(W),
      "  rank of W^T W (8 x 8) =", np.linalg.matrix_rank(W.T @ W))
