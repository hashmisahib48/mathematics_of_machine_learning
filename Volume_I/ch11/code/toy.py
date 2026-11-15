# Delivery data with a second feature, solved by coordinate descent.
import numpy as np

dist = np.array([1.0, 2.0, 3.0, 4.0, 5.0])     # km
lights = np.array([2.0, 1.0, 4.0, 3.0, 5.0])   # traffic lights
y = np.array([3.0, 4.0, 6.0, 7.0, 10.0])       # minutes
X = np.c_[dist, lights]
n = len(y)
Xc, yc = X - X.mean(axis=0), y - y.mean()      # centre first
v = (Xc ** 2).mean(axis=0)                     # v_j, Eq. 11.3-3
c = Xc.T @ yc / n                              # x_j'y / n
print("v =", v, "  c =", c, "  alpha_max =", np.abs(c).max())


def soft(z, a):                                # Eq. 11.3-6
    return np.sign(z) * max(abs(z) - a, 0.0)


alpha = 1.0
w1 = soft(c[0], alpha) / v[0]                  # distance only
b1 = y.mean() - w1 * dist.mean()
print(f"distance only: w = {w1:.4f}   b = {b1:.4f}")

w, r = np.zeros(2), yc.copy()                  # r = y_c - X_c w
for sweep in range(1, 4):
    for j in range(2):
        z = Xc[:, j] @ r / n + v[j] * w[j]     # Eq. 11.3-10
        w_new = soft(z, alpha) / v[j]
        r += Xc[:, j] * (w[j] - w_new)
        w[j] = w_new
        print(f"  sweep {sweep} j={j+1} z={z:7.4f} w={w_new:.4f}")

G = Xc.T @ Xc / n                              # Eq. 11.3-9
w_ex = np.linalg.solve(G, c - alpha)           # Eq. 11.3-11
b_ex = y.mean() - X.mean(axis=0) @ w_ex
print("both active:", np.round(w_ex, 4), " b =", round(b_ex, 4))
w_en = np.linalg.solve(G + 0.5 * np.eye(2), c - 0.5)   # Eq. 11.4-4
print("elastic net, Eq. 11.4-4:", np.round(w_en, 4))
