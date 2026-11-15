# The two worked examples of Chapter 45, checked in NumPy.
import numpy as np
np.set_printoptions(precision=4, suppress=True)

# --- Example 45.5-1: random Fourier features, D = 4, gamma = 0.5
w = np.array([1.76, 0.40, 0.98, 2.24])     # omega_j ~ N(0, 2 gamma)
b = np.array([2.66, 4.06, 2.75, 5.60])     # b_j ~ U[0, 2 pi]
c0, c1 = np.cos(b), np.cos(w + b)          # at x = 0 and x = 1
fx, fz = np.sqrt(2 / 4) * c0, np.sqrt(2 / 4) * c1   # Eq. 45.4-3
print("cos at 0:", c0, "\ncos at 1:", c1)
print("z(0):", fx, "\nz(1):", fz)
print("products", fx * fz, " sum", round(fx @ fz, 4))
kap = np.exp(-0.5)                         # exact k(0, 1)
sd = np.sqrt((1 + kap**4 / 2 - kap**2) / 4)         # Eq. 45.4-4
print("exact", kap.round(4), " std of the estimate", sd.round(4))

# --- Example 45.5-2: Nystroem, five points, landmarks 1 and 3
t = np.arange(5.0)
K = np.exp(-0.5 * (t[:, None] - t[None, :]) ** 2)
idx = [1, 3]
C, W = K[:, idx], K[np.ix_(idx, idx)]
lam, U = np.linalg.eigh(W)
Wmh = U @ np.diag(lam ** -0.5) @ U.T       # W^(-1/2), Eq. 45.4-7
print("\nlambda(W)", lam, " W^-1/2\n", Wmh)
F = C @ Wmh                                # features, Eq. 45.4-7
print("features of 0..4:\n", F[:, 0], "\n", F[:, 1])
Kh = F @ F.T                               # equals C W^-1 C^T
print("K approx:\n", Kh.round(3))
err = K - Kh
psd = np.linalg.eigvalsh(err).min() > -1e-12
print("max error", abs(err).max().round(4),
      " trace", np.trace(err).round(4), " PSD:", psd)
print("z(1).z(1) =", (F[1] @ F[1]).round(4), " z(1).z(3) =",
      (F[1] @ F[3]).round(4))
