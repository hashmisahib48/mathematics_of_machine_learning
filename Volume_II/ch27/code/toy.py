# Six lemonade days: two rounds of boosting by hand (eta = 0.5).
import numpy as np
from scipy.special import expit

x = np.array([1, 2, 3, 4, 5, 6.])          # hours of sun
y = np.array([3, 5, 4, 10, 14, 12.])       # glasses sold (dozens)
eta = 0.5
np.set_printoptions(linewidth=60)


def best_stump(x, r):
    """Threshold and leaf means that minimise the squared error."""
    best = None
    for s in (x[:-1] + x[1:]) / 2:
        a, b = r[x <= s], r[x > s]
        sse = ((a - a.mean()) ** 2).sum() \
            + ((b - b.mean()) ** 2).sum()
        if best is None or sse < best[0]:
            best = (sse, s, a.mean(), b.mean())
    return best


F = np.full(6, y.mean())                   # F_0: the mean
print("F0 =", F[0], " MSE", ((y - F) ** 2).mean().round(4))
for m in (1, 2):
    r = y - F                              # residuals, Eq. 27.4-1
    sse, s, cl, cr = best_stump(x, r)
    F = F + eta * np.where(x <= s, cl, cr)  # Eq. 27.3-6
    print(f"round {m}: r = {r}, split {s},"
          f" leaves {cl:.2f} {cr:.2f}")
    print(f"         F = {F}, MSE {((y - F) ** 2).mean():.4f}")

# Absolute error: tree on the signs, leaves = residual medians.
F = np.full(6, np.median(y))
r = y - F
left = x <= 3.5
print("\nabsolute error: F0 =", F[0], " signs", np.sign(r))
cl, cr = np.median(r[left]), np.median(r[~left])
F1 = F + eta * np.where(left, cl, cr)
print("leaf medians", cl, cr, " F1 =", F1)
