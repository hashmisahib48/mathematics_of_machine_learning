# Problem 6B.1: Listing 6.2 with 100 training samples instead of 15.
import numpy as np
from numpy.polynomial import Polynomial


def run(n_train):
    rng = np.random.default_rng(7)                  # same generator
    def draw(n):
        x = np.sort(rng.uniform(0, 1, n))
        return x, np.sin(2 * np.pi * x) + rng.normal(0, 0.25, n)
    x_tr, y_tr = draw(n_train)
    x_te, y_te = draw(400)                          # 400 test samples
    out = []
    for d in range(15):                             # degrees 0 ... 14
        f = Polynomial.fit(x_tr, y_tr, d)           # least squares
        out.append((np.sqrt(np.mean((y_tr - f(x_tr)) ** 2)),
                    np.sqrt(np.mean((y_te - f(x_te)) ** 2))))
    return np.array(out)


big, small = run(100), run(15)
print("        n = 100          n = 15")
print("degree  train  test    train  test")
for d in range(15):
    print(f"{d:6d} {big[d, 0]:6.3f} {big[d, 1]:6.3f}"
          f" {small[d, 0]:7.3f} {small[d, 1]:9.3f}")
for n, r in ((100, big), (15, small)):
    b = int(np.argmin(r[:, 1]))
    print(f"n = {n:3d}: best degree {b}, train {r[b, 0]:.3f}, "
          f"test {r[b, 1]:.3f}; degree 14: {r[14, 1]:.3f}")
p, s2 = 15, 0.0625                       # Eq. 6.4-4 for degree 14
gap = big[14, 1] ** 2 - big[14, 0] ** 2
print(f"n = 100, degree 14: gap in MSE {gap:.4f}, "
      f"theory 2 p sigma^2 / n = {2 * p * s2 / 100:.4f}")
