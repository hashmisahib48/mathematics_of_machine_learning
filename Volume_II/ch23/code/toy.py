# Delivery data, linear SVR with epsilon = 1 and C = 1, by hand.
import numpy as np
from sklearn.svm import SVR

x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y = np.array([3.0, 4.0, 6.0, 7.0, 10.0])
eps, C = 1.0, 1.0
i, j = 1, 4              # guess: samples 2 and 5 are on the edges
# unknowns (w, b, beta), Eq. 23.5-1; beta_j = -beta_i = beta
A = np.array([[x[i], 1.0, 0.0],             # w x_i + b = y_i + eps
              [x[j], 1.0, 0.0],             # w x_j + b = y_j - eps
              [-1.0, 0.0, x[j] - x[i]]])    # w = beta (x_j - x_i)
rhs = np.array([y[i] + eps, y[j] - eps, 0.0])
w, b, beta = np.linalg.solve(A, rhs)
print(f"w = {w:.4f}  b = {b:.4f}  beta = {beta:.4f}")

bv = np.zeros(5)
bv[i], bv[j] = -beta, beta
r = y - (w * x + b)                         # residuals
print("r =", np.round(r, 3))
print("all inside:", bool((np.abs(r) <= eps + 1e-12).all()))
print("sum beta =", round(bv.sum(), 12), " w =", round(bv @ x, 4))

# primal (Eq. 23.3-1) and dual (Eq. 23.3-5) values
primal = 0.5 * w**2 + C * np.maximum(0, np.abs(r) - eps).sum()
dual = -0.5 * (bv @ x)**2 + y @ bv - eps * np.abs(bv).sum()
print(f"primal = {primal:.4f}  dual = {dual:.4f}")
print(f"f(6) = {w * 6 + b:.4f}")

m = SVR(kernel="linear", C=C, epsilon=eps, tol=1e-10)
m.fit(x[:, None], y)
print("SVR:", m.coef_[0, 0].round(4), m.intercept_[0].round(4),
      m.support_, m.dual_coef_.round(4))
