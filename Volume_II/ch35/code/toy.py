# PLS1 by hand on four samples and two readings (Example 35.5-1).
import numpy as np

X = np.array([[-3., 1], [-2, -1], [2, 1], [3, -1]])  # centred
y = np.array([0., -1, 2, -1])                        # centred fat

Xk, yk = X.copy(), y.copy()
W, P, q, T = [], [], [], []
for k in (1, 2):
    g = Xk.T @ yk                      # X^T y, Eq. 35.3-1
    w = g / np.linalg.norm(g)          # unit weight vector
    w = w * np.sign(w[np.argmax(abs(w))])   # sign rule
    t = Xk @ w                         # scores, Eq. 35.3-3
    p = Xk.T @ t / (t @ t)             # loading of X, Eq. 35.3-4
    c = yk @ t / (t @ t)               # loading of y, Eq. 35.3-5
    print(f"k={k}  Xty={g.round(4)}  w={w.round(4)}")
    print(f"     t={t.round(4)}  t.t={t @ t:.4f}")
    print(f"     p={p.round(4)}  q={c:.4f}")
    Xk = Xk - np.outer(t, p)           # deflation, Eq. 35.3-6
    yk = yk - c * t
    print(f"     deflated y={yk.round(4)}")
    W.append(w); P.append(p); q.append(c); T.append(t)
W, P, q = np.array(W).T, np.array(P).T, np.array(q)
R = W @ np.linalg.inv(P.T @ W)         # rotations, Eq. 35.3-8
print("R =", R.round(4).tolist())
for A in (1, 2):
    B = R[:, :A] @ q[:A]               # coefficients, Eq. 35.3-9
    sse = ((y - X @ B) ** 2).sum()
    print(f"A={A}: B={B.round(4)}  SSE={sse:.4f}")
print("OLS:", np.linalg.solve(X.T @ X, X.T @ y).round(4))
print("scores orthogonal:", round(float(T[0] @ T[1]), 12))
