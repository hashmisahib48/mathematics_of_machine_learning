# Problem 2B.2: a column that repeats another one in other units
import numpy as np

km = np.array([1.0, 2.0, 3.0, 4.0, 5.0])        # distances in km
X = np.column_stack([np.ones(5), km,
                     0.621371 * km])   # ones, km, miles
print("X =\n", X)
s = np.linalg.svd(X, compute_uv=False)            # singular values
print("singular values:", s)
print("rank from matrix_rank:", np.linalg.matrix_rank(X))
eps = np.finfo(float).eps
tol = s.max() * max(X.shape) * eps       # numpy's threshold
print("tolerance used by matrix_rank:", tol)

U, s, Vt = np.linalg.svd(X)
v3 = Vt[2]                      # weakest direction
print("third right singular vector:", np.round(v3, 4))
print("X @ v3 =", np.round(X @ v3, 15))
print("norm of X @ v3:", np.linalg.norm(X @ v3))
w = np.array([0.0, 0.621371, -1.0])   # 0.621371 km - miles
print("X @ (0, 0.621371, -1) =", X @ w)
print("X^T X has rank", np.linalg.matrix_rank(X.T @ X),
      " cond(X^T X) = %.2e" % np.linalg.cond(X.T @ X))
