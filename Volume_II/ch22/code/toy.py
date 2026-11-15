# Eight seeds: the hard-margin problem solved through its dual.
import numpy as np
from sklearn.svm import SVC

X = np.array([[1, 5], [2, 3], [3, 4], [4, 1],     # kind A, y = -1
              [7, 6], [8, 2], [4, 7], [6, 3]], float)  # B, y = +1
y = np.array([-1, -1, -1, -1, 1, 1, 1, 1])
sv = [2, 6, 7]                   # support set, from Figure 22.1-1

K = X[sv] @ X[sv].T              # Gram matrix of the three seeds
M = np.zeros((4, 4))             # unknowns: alpha_1..alpha_3 and b
M[:3, :3] = K * y[sv]            # f(x_k) = y_k, that is
M[:3, 3] = 1                     # sum a_i y_i K_ik + b = y_k
M[3, :3] = y[sv]                 # sum a_i y_i = 0 (Eq. 22.3-6)
a1, a2, a3, b = np.linalg.solve(M, np.r_[y[sv], 0.0])
alpha = np.zeros(8)
alpha[sv] = [a1, a2, a3]
w = (alpha * y) @ X              # Eq. 22.3-6
print("alpha (seeds 3, 7, 8):", np.round(alpha[sv], 4))
print("w =", w, " b =", round(b, 4))
print("|w|^2 =", round(w @ w, 4), " sum alpha =", alpha.sum())
print("margin 2/|w| =", round(2 / np.linalg.norm(w), 4))
f = X @ w + b
print("f(x), seeds 1-8:", np.round(f, 2))
print("all y f(x) >= 1:", bool(np.all(y * f >= 1 - 1e-12)))

m = SVC(kernel="linear", C=1e6, tol=1e-10).fit(X, y)
print("SVC support_ + 1:", m.support_ + 1)
print("SVC dual_coef_  :", np.round(m.dual_coef_[0], 4))
print("SVC coef_, b    :", m.coef_[0], m.intercept_)
