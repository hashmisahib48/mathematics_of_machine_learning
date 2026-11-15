# Problem 3B.2: second-order Taylor expansion of softplus at x = 1
import numpy as np

sigma = lambda z: 1 / (1 + np.exp(-z))
softplus = lambda z: np.log1p(np.exp(z))

x0, x = 1.0, 1.5
h = x - x0                            # the step
f0 = softplus(x0)                     # value
f1 = sigma(x0)                        # slope, Eq. 3.2-3
f2 = sigma(x0) * (1 - sigma(x0))      # curvature, Eq. 3.2-2
print(f"softplus(1) = {f0:.4f}  slope = {f1:.4f}"
      f"  curvature = {f2:.4f}")

first = f0 + f1 * h                   # Eq. 3.9-1
second = first + 0.5 * f2 * h ** 2    # Eq. 3.9-2
exact = softplus(x)
print(f"exact value at 1.5 : {exact:.4f}")
print(f"first order  : {first:.4f}   error {exact - first:+.5f}")
print(f"second order : {second:.4f}   error {exact-second:+.5f}")

# the rest: f''' h^3 / 6, f''' = s(1-s)(1-2s)
f3 = f2 * (1 - 2 * sigma(x0))
print(f"f''' = {f3:.4f}   predicted error {f3 * h ** 3 / 6:+.5f}")
for h in (0.1, 0.25, 0.5, 1.0):       # the error falls like h^3
    e = softplus(x0 + h) - (f0 + f1 * h + 0.5 * f2 * h ** 2)
    print(f"h = {h:4.2f}  error {e:+.6f}  error/h^3 {e/h**3:+.4f}")
