# Problem 13B.2: the L1 penalty of QuantileRegressor
import numpy as np
from sklearn.linear_model import QuantileRegressor

x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y = np.array([3.0, 4.0, 6.0, 7.0, 20.0])    # y5 = 20 is wrong
X = x[:, None]

def pinball(r, tau=0.5):                     # Eq. 13.5-1, averaged
    return np.mean(np.where(r >= 0, tau * r, (tau - 1) * r))

print("alpha  slope intercept  loss  alpha*|w|  objective")
for a in (0.0, 0.01, 0.1, 0.5, 1.0):
    q = QuantileRegressor(alpha=a).fit(X, y)     # tau = 0.5, HiGHS
    w, b = q.coef_[0], q.intercept_
    loss = pinball(y - b - w * x)
    print(f"{a:5.2f} {w:7.3f} {b:9.3f} {loss:6.3f} {a * abs(w):9.3f}"
          f" {loss + a * abs(w):10.3f}")

# Eq. 13.5-3 for three candidate lines (slope, intercept)
cand = {"(1.5, 1.5)": (1.5, 1.5), "(1.333, 1.667)": (4 / 3, 5 / 3),
        "(0, 6)": (0.0, 6.0)}
print("\nobjective of each candidate line:")
print("alpha  " + "  ".join(f"{k:>14s}" for k in cand))
for a in (0.0, 0.1, 0.5, 1.0):
    row = [pinball(y - b - w * x) + a * abs(w)
           for w, b in cand.values()]
    print(f"{a:5.2f}  " + "  ".join(f"{v:14.4f}" for v in row))
