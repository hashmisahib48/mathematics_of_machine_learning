# Problem 3B.3: central difference of sin at x = 1
import numpy as np

x, exact = 1.0, np.cos(1.0)
eps = np.finfo(float).eps             # machine precision, 2.2e-16
print("h        slope            error    h^2/6|cos 1|   eps/h")
best = (None, np.inf)
for k in range(1, 15):
    h = 10.0 ** -k
    slope = (np.sin(x + h) - np.sin(x - h)) / (2 * h)  # 3.10-1
    err = abs(slope - exact)
    trunc = h ** 2 / 6 * abs(np.cos(x))   # first term left out
    print(f"1e-{k:<2d} {slope:.12f} {err:9.2e} {trunc:11.2e}"
          f" {eps / h:10.2e}")
    if err < best[1]:
        best = (k, err)
print(f"\nbest step 1e-{best[0]}, error {best[1]:.2e}")
print(f"cos(1) = {exact:.12f}")
# truncation h^2 |cos 1| / 6 equals rounding eps sin(1) / h at
hopt = (3 * eps * np.sin(x) / abs(np.cos(x))) ** (1 / 3)
print(f"balance of the two errors: h = {hopt:.1e}")
