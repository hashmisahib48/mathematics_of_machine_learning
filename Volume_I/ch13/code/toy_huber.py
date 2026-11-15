# Huber fit of the delivery data with one wrong entry (y5 = 20).
import numpy as np
from sklearn.linear_model import HuberRegressor

x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y = np.array([3.0, 4.0, 6.0, 7.0, 20.0])
eps, n = 1.35, 5

h = HuberRegressor(epsilon=eps, alpha=0.0).fit(x[:, None], y)
b, w = h.intercept_, h.coef_[0]
r = y - b - w * x                          # residuals, Eq. 13.3-2
out = np.abs(r) > eps * h.scale_           # |r| > eps * sigma
k = out.sum()
sig2 = np.sum(r[~out] ** 2) / (n - k * eps ** 2)       # Eq. 13.3-8
sigma = np.sqrt(sig2)
omega = np.minimum(1.0, eps * sigma / np.abs(r))       # Eq. 13.3-4
print("b, w     =", np.round([b, w], 4))
print("residual =", np.round(r, 4))
print("sigma    =", round(sigma, 4), "scale_ =", round(h.scale_, 4))
print("eps*sigma=", round(eps * sigma, 4), "  outliers:", k)
print("omega    =", np.round(omega, 4))

# weighted normal equations, Eq. 13.3-6
S = [omega.sum(), (omega * x).sum(), (omega * x * x).sum(),
     (omega * y).sum(), (omega * x * y).sum()]
print("sums     =", np.round(S, 4))
A = np.array([[S[0], S[1]], [S[1], S[2]]])
print("solve    =", np.round(np.linalg.solve(A, S[3:]), 4))
