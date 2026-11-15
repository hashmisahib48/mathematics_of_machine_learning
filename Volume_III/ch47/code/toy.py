# KDE of four waits at x0 = 58: Gaussian (h = 4), compact (h = 8).
import numpy as np

x = np.array([52.0, 56.0, 80.0, 84.0])      # waits, minutes
n, x0 = len(x), 58.0
u = (x0 - x) / 4.0                          # scaled distances
K = np.exp(-u**2 / 2) / np.sqrt(2 * np.pi)  # Gaussian K(u)
f = K.sum() / (n * 4.0)                     # Eq. 47.3-2 with p = 1
print("u =", u, " K(u) =", K.round(4))
print(f"gaussian     f(58) = {f:.5f}  ln f = {np.log(f):.4f}")

u = np.abs(x0 - x) / 8.0                    # compact kernels, h = 8
epan = np.where(u < 1, 0.75 * (1 - u**2), 0.0)
top = np.where(u < 1, 0.5, 0.0)
print(f"epanechnikov f(58) = {epan.sum() / (n * 8.0):.5f}")
print(f"tophat       f(58) = {top.sum() / (n * 8.0):.5f}")
