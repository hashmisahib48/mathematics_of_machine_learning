# NIPALS inner loop for two targets (Example 35.5-2), first component.
import numpy as np

X = np.array([[-3., 1], [-2, -1], [2, 1], [3, -1]])
Y = np.array([[0., 1], [-1, 0], [2, -1], [-1, 0]])   # fat, protein

u = Y[:, 0].copy()                  # start: first column of Y
w_old = np.zeros(2)
print(" it      w1      w2      c1      c2   |w-w_old|^2")
for it in range(1, 20):
    w = X.T @ u / (u @ u)           # Eq. 35.4-4
    w /= np.linalg.norm(w)
    t = X @ w
    c = Y.T @ t / (t @ t)           # Eq. 35.4-5
    u = Y @ c / (c @ c)             # Eq. 35.4-6
    d = ((w - w_old) ** 2).sum()
    print(f"{it:3d}{w[0]:8.4f}{w[1]:8.4f}{c[0]:8.4f}{c[1]:8.4f}{d:14.2e}")
    if d < 1e-6:
        break
    w_old = w
U, s, Vt = np.linalg.svd(X.T @ Y)
print("X^T Y =", (X.T @ Y).tolist())
print("SVD: u1 =", U[:, 0].round(4), " sigma =", s.round(4))
print("t =", t.round(4), " c =", c.round(4))
