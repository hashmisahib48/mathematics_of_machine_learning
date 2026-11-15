# SMO on the eight seeds: every step printed (Section 22.7).
import numpy as np

X = np.array([[1, 5], [2, 3], [3, 4], [4, 1],
              [7, 6], [8, 2], [4, 7], [6, 3]], float)
s = np.array([-1, -1, -1, -1, 1, 1, 1, 1.])
K, C = X @ X.T, 10.0                   # linear kernel, C large
a = np.zeros(8)
W = lambda a: a.sum() - 0.5 * (a * s) @ K @ (a * s)  # Eq. 22.3-9

print("step i j    q  Ei-Ej    L     H   a_i     a_j      W")
for step in range(1, 40):
    E = K @ (a * s) - s                # E_k = f(x_k) - y_k, no b
    v = -E                             # violation of sample k
    up = ((s > 0) & (a < C)) | ((s < 0) & (a > 0))
    lo = ((s > 0) & (a > 0)) | ((s < 0) & (a < C))
    i = np.flatnonzero(up)[np.argmax(v[up])]
    if v[i] - v[lo].min() < 1e-3:
        break
    q = np.maximum(K[i, i] + np.diag(K) - 2 * K[i], 1e-12)
    j = np.argmax(np.where(lo & (v < v[i]), (v[i] - v)**2 / q, 0))
    if s[i] != s[j]:
        L, H = max(0, a[j] - a[i]), min(C, C + a[j] - a[i])
    else:
        L, H = max(0, a[i] + a[j] - C), min(C, a[i] + a[j])
    aj = a[j] + s[j] * (E[i] - E[j]) / q[j]   # Eq. 22.6-3
    aj = np.clip(aj, L, H)
    ai = a[i] + s[i] * s[j] * (a[j] - aj)  # Eq. 22.6-4
    a[i], a[j] = ai, aj
    if step <= 9:
        print(f"{step:4d} {i+1} {j+1} {q[j]:4.0f} {E[i]-E[j]:6.2f}"
              f" {L:4.2f} {H:5.2f} {ai:.4f} {aj:.4f} {W(a):.4f}")
w = (a * s) @ X
print("steps:", step - 1, " alpha =", np.round(a, 4))
print("W =", round(W(a), 4), " |w|^2/2 =", round(w @ w / 2, 4))
