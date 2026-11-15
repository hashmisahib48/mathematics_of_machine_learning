# Two readings, one question: Examples 30.6-1 and 30.6-2 in NumPy.
import numpy as np

x = np.array([0.0, 2.0])             # settings
y = np.array([1.2, 0.4])             # readings
sn2 = 0.1                            # noise variance sigma^2


def kern(a, b, ell=1.0):             # RBF with sigma_f = 1
    return np.exp(-(a[:, None] - b[None, :])**2 / (2 * ell**2))


S = kern(x, x) + sn2 * np.eye(2)     # Eq. 30.3-3
L = np.linalg.cholesky(S)            # S = L L^T
a = np.linalg.solve(L.T, np.linalg.solve(L, y))
print("S =", np.round(S, 5).tolist(), "  a =", np.round(a, 4))

for xs in (1.0, 4.0):                # Eq. 30.3-7 at two settings
    ks = kern(x, np.array([xs]))[:, 0]
    v = np.linalg.solve(L, ks)
    mean, var = ks @ a, 1.0 - v @ v
    print(f"x* = {xs}: k* = {np.round(ks, 5)}  mean = {mean:.4f}  "
          f"var = {var:.4f}  sd = {np.sqrt(var):.4f}")

print("ell    fit     complexity  constant   total")
for ell in (0.3, 1.0, 3.0):          # the three terms of Eq. 30.4-2
    S = kern(x, x, ell) + sn2 * np.eye(2)
    fit = -0.5 * y @ np.linalg.solve(S, y)
    cx = -0.5 * np.log(np.linalg.det(S))
    const = -np.log(2 * np.pi)       # -(n/2) ln(2 pi), n = 2
    print(f"{ell:3.1f}  {fit:7.4f}  {cx:9.4f}  {const:9.4f}  "
          f"{fit + cx + const:8.4f}")
