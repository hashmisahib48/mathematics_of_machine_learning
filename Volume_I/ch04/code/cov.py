# Covariance, correlation and the multivariate normal (§4.6).
import numpy as np

J = np.array([[0.4, 0.1], [0.1, 0.4]])       # P(X=x, Y=y)
vals = np.array([0.0, 1.0])
mx, my = vals @ J.sum(axis=1), vals @ J.sum(axis=0)    # marginals
cov = vals @ J @ vals - mx * my                        # Eq. 4.6-1
vx = (vals - mx) ** 2 @ J.sum(axis=1)
print(f"joint table: Cov = {cov:.2f}  rho = {cov / vx:.2f}")

x = np.array([1.0, 2, 3, 4, 5])              # delivery data
y = np.array([3.0, 4, 6, 7, 10])
print("covariance matrix:\n", np.cov(x, y))  # divides by n - 1
r = np.corrcoef(x, y)[0, 1]                  # Eq. 4.6-2
print(f"correlation = {r:.4f}   r^2 = {r**2:.4f}")

Sigma = np.array([[4.0, 3.0], [3.0, 5.0]])   # a multivariate normal
rng = np.random.default_rng(1)
S = rng.multivariate_normal([0, 0], Sigma, size=100_000)
print("sample covariance:\n", np.round(np.cov(S.T), 2))
vals, vecs = np.linalg.eigh(Sigma)           # ellipse axes
print("eigenvalues of Sigma:", np.round(vals, 4))
print("1-sd half-axes:", np.round(np.sqrt(vals), 4))
a = np.array([1.0, 1.0])                     # the sum X1 + X2
print("Var(a^T x) = a^T Sigma a:", a @ Sigma @ a,
      "  sample:", round(np.var(S @ a, ddof=1), 2))
