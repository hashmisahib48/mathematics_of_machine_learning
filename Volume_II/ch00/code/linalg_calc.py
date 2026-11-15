# Vectors, matrices and gradients: checks of Examples R.2-1, R.3-1.
import numpy as np

a, b = np.array([2., 1.]), np.array([3., 4.])
c = a @ b / (a @ a)                          # Eq. 2.4-2 (Volume I)
p, r = c * a, b - c * a
print("c =", c, " p =", p, " r =", r, " a.r =", a @ r)

A = np.array([[2., 1.], [1., 0.], [0., 1.]])
S = A.T @ A
lam, Q = np.linalg.eigh(S)                   # ascending order
print("A^T A =", S.tolist(), " eigenvalues:", lam.round(4))
q6 = Q[:, 1] * np.sign(Q[0, 1])              # sign is arbitrary
print("eigenvector of 6:", q6.round(4), " S q - 6 q:",
      np.abs(S @ q6 - 6 * q6).max().round(12))
s = np.linalg.svd(A, compute_uv=False)
print("singular values:", s.round(4),
      " cond:", round(s[0] / s[1], 4))
yy = np.array([1., 2., 3.])
print("projection of (1,2,3):", A @ np.linalg.solve(S, A.T @ yy))

X = np.c_[np.ones(5), np.arange(1., 6)]      # Chapter 9 data
y = np.array([3., 4, 6, 7, 10])
J = lambda w: np.sum((y - X @ w) ** 2)
grad = lambda w: -2 * X.T @ (y - X @ w)      # Eq. 3.7-6
H = 2 * X.T @ X                              # Eq. 3.8-4
w0 = np.zeros(2); g = grad(w0)
print("J(0) =", J(w0), " gradient:", g)
print("Hessian:", H.tolist())
h = 1e-5                                     # Eq. 3.10-1
num = [(J(w0 + h * e) - J(w0 - h * e)) / (2 * h) for e in np.eye(2)]
print("numerical gradient:", np.round(num, 3))
w1 = w0 - 0.01 * g                           # one gradient step
print("GD step:", w1, " J =", round(J(w1), 2))
wn = w0 - np.linalg.solve(H, g)              # one Newton step
print("Newton step:", wn.round(4), " gradient:", grad(wn).round(8))
