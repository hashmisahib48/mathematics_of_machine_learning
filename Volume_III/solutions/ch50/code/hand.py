# Hand checks for Problems 50A.1 and 50A.2 (NumPy)
import numpy as np

# ---- 50A.1
cat = np.array(list("aabccc")); y = np.array([4, 6, 10, 3, 5, 7.])
ks = np.unique(cat); D = (cat[:, None] == ks).astype(float)
print("D =\n", D.astype(int))
print("D^T D =\n", (D.T @ D).astype(int))
print("D^T y =", D.T @ y, "  n_k =", D.sum(0))
w = (D.T @ y) / D.sum(0)                         # Eq. 50.3-1
print("weights w =", w, " (category means)")
A = np.hstack([np.ones((6, 1)), D])
print("[1 | D] columns:", A.shape[1], " rank:", np.linalg.matrix_rank(A), " 1 - d_a - d_b - d_c =", A[:, 0] - D.sum(1))
print("det of [1|D]^T [1|D]: %.3g" % np.linalg.det(A.T @ A))
B = np.hstack([np.ones((6, 1)), D[:, 1:]])        # drop category a
beta = np.linalg.solve(B.T @ B, B.T @ y)
print("drop a: rank", np.linalg.matrix_rank(B), " columns", B.shape[1], " [intercept, b, c] =", beta.round(4),
      " fitted values", (B @ beta).round(4), " same as D w:", np.allclose(B @ beta, D @ w))
print("B^T B =\n", (B.T @ B).astype(int))

# ---- 50A.2
n = np.array([1, 4, 20.]); mc = np.array([30, 20, 10.]); yb, m = 14.0, 4.0
lam = n / (n + m); t = lam * mc + (1 - lam) * yb
print("\nlambda = n/(n+m):", lam.round(4), " fractions 1/5, 4/8, 20/24")
print("t =", t.round(4), " check (n*mean + m*ybar)/(n+m):", ((n * mc + m * yb) / (n + m)).round(4))
print("change t - mean:", (t - mc).round(4), "  moved by", (np.abs(t - mc) / np.abs(mc - yb)).round(3), "of the distance to the overall mean")
# the minimiser of Eq. 50.4-1 by brute force for the first category: n rows of target 30, m imaginary rows of target 14
g = np.linspace(10, 30, 200001); f = n[0] * (mc[0] - g) ** 2 + m * (g - yb) ** 2
print("brute-force minimiser for the one-row category: %.4f" % g[f.argmin()])
print("the usual slip, lambda and 1 - lambda swapped:", ((1 - lam) * mc + lam * yb).round(3))
