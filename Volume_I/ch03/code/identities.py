# Gradient identities and the Hessian, checked on small matrices.
import numpy as np

def num_grad(f, w, h=1e-6):                # central differences
    g = np.zeros_like(w)
    for j in range(len(w)):
        e = np.zeros_like(w); e[j] = h
        g[j] = (f(w + e) - f(w - e)) / (2 * h)
    return g

a = np.array([2.0, -1.0, 3.0])
A = np.array([[2.0, 1, 0], [1, 3, 1], [0, 1, 4]])   # symmetric
Bm = np.array([[1.0, 2, 0], [0, 3, 1], [4, 1, 2]])  # not symmetric
w = np.array([1.0, 2.0, -1.0])
print("grad of a.w     :",
      np.round(num_grad(lambda v: a @ v, w), 4))
print("  formula a     :", a)
print("grad of w'Aw    :",
      np.round(num_grad(lambda v: v @ A @ v, w), 4))
print("  formula 2Aw   :", 2 * A @ w)
print("grad of w'Bw    :",
      np.round(num_grad(lambda v: v @ Bm @ v, w), 4))
print("  formula (B+B')w:", (Bm + Bm.T) @ w)

x = np.array([1.0, 2, 3, 4, 5]); y = np.array([3.0, 4, 6, 7, 10])
X = np.c_[np.ones(5), x]                  # column of ones
J = lambda v: np.sum((y - X @ v) ** 2)
v = np.array([0.0, 1.0])                             # (b, w)
print("\ngradient of J at (0, 1):", -2 * X.T @ (y - X @ v))
print("  by numbers           :", np.round(num_grad(J, v), 4))

H = 2 * X.T @ X                           # Eq. 3.8-4
print("Hessian 2 X'X:\n", H)
lam = np.linalg.eigvalsh(H)
print("eigenvalues:", np.round(lam, 3),
      "  ratio:", round(lam[1] / lam[0], 1))
