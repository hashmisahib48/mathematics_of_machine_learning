# Three unit directions for the four samples: variance, correlation,
# covariance of the score t = X w with the target (Section 35.2).
import numpy as np

X = np.array([[-3., 1], [-2, -1], [2, 1], [3, -1]])
y = np.array([0., -1, 2, -1])

lam, V = np.linalg.eigh(X.T @ X)
pca = V[:, -1] * np.sign(V[0, -1])                   # largest variance
pls = X.T @ y / np.linalg.norm(X.T @ y)              # Eq. 35.3-1
b = np.linalg.solve(X.T @ X, X.T @ y)
ols = b / np.linalg.norm(b)                          # largest correlation
print(f"{'direction':8s}{'w1':>8s}{'w2':>8s}{'t.t':>8s}"
      f"{'corr':>8s}{'t.y':>8s}{'angle':>8s}")
for name, w in (("PCA", pca), ("PLS", pls), ("OLS", ols)):
    t = X @ w
    r = t @ y / np.sqrt((t @ t) * (y @ y))
    a = np.degrees(np.arccos(min(1.0, abs(w @ pca))))
    print(f"{name:8s}{w[0]:8.4f}{w[1]:8.4f}{t @ t:8.3f}"
          f"{r:8.4f}{t @ y:8.4f}{a:8.1f}")
print("eigenvalues of X^T X:", lam.round(4))
