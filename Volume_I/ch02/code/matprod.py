# Products, transpose and the shapes rule (Section 2.5).
import numpy as np

M = np.array([[1.0, 2.0], [3.0, 4.0]])
N = np.array([[2.0, 0.0], [1.0, 3.0]])
x = np.array([1.0, 1.0])
print("M @ N =", (M @ N).tolist())                # Eq. 2.5-2
print("N @ M =", (N @ M).tolist())
print("M * N =", (M * N).tolist(), "(entry by entry, no product)")
print("M @ x =", M @ x, "  M^T =", M.T.tolist())  # Eq. 2.5-1
print("(MN)^T = N^T M^T:", np.allclose((M @ N).T, N.T @ M.T))

R = np.arange(6.0).reshape(2, 3)                  # a 2 x 3 table
print("shapes:", R.shape, "@", R.T.shape, "->", (R @ R.T).shape)
print("shapes:", R.T.shape, "@", R.shape, "->", (R.T @ R).shape)
try:
    R @ R
except ValueError as e:
    print("ValueError:", str(e)[:60])
S = R.T @ R
print("R^T R is symmetric:", np.allclose(S, S.T))
