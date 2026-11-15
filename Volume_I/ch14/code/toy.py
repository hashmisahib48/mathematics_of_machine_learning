# Poisson regression by Newton steps on the crack-count data.
import numpy as np

x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])   # load cycles, 10^5
y = np.array([1.0, 2.0, 2.0, 4.0, 6.0])   # cracks found
X = np.column_stack([np.ones(5), x])      # ones column, Eq. 9.4-2


def deviance(y, mu):                      # Eq. 14.3-6, summed
    return np.sum(2 * (y * np.log(y / mu) - (y - mu)))


beta = np.array([np.log(y.mean()), 0.0])  # start: mu = mean of y
for k in range(5):
    mu = np.exp(X @ beta)                 # inverse link, Eq. 14.3-1
    g = X.T @ (y - mu)                    # minus n times gradient
    H = X.T @ (mu[:, None] * X)           # n x Hessian, Eq. 14.3-12
    print(f"step {k}: b = {beta[0]:.4f}  w = {beta[1]:.4f}  "
          f"D = {deviance(y, mu):.5f}  max|g| = {abs(g).max():.1e}")
    beta = beta + np.linalg.solve(H, g)   # Newton step, Eq. 14.3-13

mu = np.exp(X @ beta)
D_null = deviance(y, np.full(5, y.mean()))
print(f"balance: sum(y - mu) = {np.sum(y - mu):.1e}, "
      f"sum(x(y - mu)) = {np.sum(x * (y - mu)):.1e}")
print(f"D = {deviance(y, mu):.4f}   D_null = {D_null:.4f}   "
      f"D2 = {1 - deviance(y, mu) / D_null:.4f}")
