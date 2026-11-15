# Ten loan applicants: every candidate split of the root, by Gini.
import numpy as np

age = np.array([35, 27, 22, 48, 33, 24, 52, 29, 41, 58])
inc = np.array([22, 40, 55, 28, 62, 18, 36, 26, 48, 75])
y = np.array([0, 1, 0, 0, 1, 0, 1, 0, 1, 1])      # 1 = repaid


def gini(labels):                        # Eq. 24.3-3
    p = np.bincount(labels, minlength=2) / len(labels)
    return 1.0 - np.sum(p ** 2)


def candidates(x, y):
    """Midpoints between sorted values and the weighted Gini."""
    v = np.unique(x)
    for t in (v[:-1] + v[1:]) / 2:
        L, R = y[x <= t], y[x > t]
        g = (len(L) * gini(L) + len(R) * gini(R)) / len(y)
        yield t, np.bincount(L, minlength=2), g       # Eq. 24.3-5


print("root impurity", gini(y))
for name, x in (("income", inc), ("age", age)):
    t, left, g = min(candidates(x, y), key=lambda c: c[2])
    print(f"{name:6s} best <= {t:4.1f}  G = {g:.4f}")
t, left, g = min(candidates(inc, y), key=lambda c: c[2])
print("decrease", round(gini(y) - g, 4))
R, a = y[inc > t], age[inc > t]   # the six on the right
print("right node: n =", len(R), " Gini", round(gini(R), 4))
s2, l2, g2 = min(candidates(a, R), key=lambda c: c[2])
print(f"  age <= {s2:4.1f}  G = {g2:.4f}")
