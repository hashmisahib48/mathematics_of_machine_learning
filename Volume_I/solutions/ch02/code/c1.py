# Problem 2C.1: 100 samples, 120 features
import numpy as np
rng = np.random.default_rng(0)
n, p = 100, 120
X = rng.normal(size=(n, p)); y = rng.normal(size=n)       # any target at all
print("rank of X (100 x 120):", np.linalg.matrix_rank(X))
G = X.T @ X
print("rank of X^T X (120 x 120):", np.linalg.matrix_rank(G), " smallest |eigenvalue|: %.1e" % np.abs(np.linalg.eigvalsh(G)).min())
print("dimension of the null space of X: p - rank =", p - np.linalg.matrix_rank(X))
w0 = np.linalg.pinv(X) @ y                               # minimum-norm solution
print("training residual of the minimum-norm weights: %.1e" % np.linalg.norm(y - X @ w0), " (R^2 = 1 on pure noise)")
U, s, Vt = np.linalg.svd(X); N = Vt[n:].T                  # basis of the null space (120 x 20)
print("null space basis shape:", N.shape, " |X N| = %.1e" % np.abs(X @ N).max())
for k in range(3):
    w = w0 + N @ rng.normal(size=N.shape[1])
    print(f"another solution {k + 1}: residual {np.linalg.norm(y - X @ w):.1e}  norm {np.linalg.norm(w):.3f}  (minimum norm {np.linalg.norm(w0):.3f})")
Xt = rng.normal(size=(1000, p)); yt = rng.normal(size=1000)       # new noise samples
print("R^2 of the interpolating fit on 1000 new samples: %.3f" % (1 - ((yt - Xt @ w0) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()))
try:
    wn = np.linalg.solve(G, X.T @ y)          # the normal equations
    print("solve returned weights of norm %.2e (condition number %.1e), residual %.2f" % (np.linalg.norm(wn), np.linalg.cond(G), np.linalg.norm(y - X @ wn)))
except np.linalg.LinAlgError as e: print("np.linalg.solve on X^T X raises:", e)
