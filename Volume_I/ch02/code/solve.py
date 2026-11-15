# solve versus inv on a badly conditioned matrix (Hilbert matrix).
import numpy as np
from scipy.linalg import hilbert

print(" n      cond(H)   err solve     err inv    "
      "resid solve   resid inv")
for n in (6, 8, 10):
    H = hilbert(n)                # entries 1 / (i + j + 1)
    x_true = np.ones(n)
    b = H @ x_true
    x1 = np.linalg.solve(H, b)    # factorise and substitute
    x2 = np.linalg.inv(H) @ b     # form the inverse, then multiply
    print(f"{n:2d}  {np.linalg.cond(H):11.3e}"
          f"  {np.abs(x1 - x_true).max():10.2e}"
          f"  {np.abs(x2 - x_true).max():10.2e}"
          f"  {np.linalg.norm(H @ x1 - b):10.2e}"
          f"  {np.linalg.norm(H @ x2 - b):10.2e}")
