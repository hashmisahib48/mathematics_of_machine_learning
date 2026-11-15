# Hand checks for the Chapter 25 solutions (25A.1 - 25A.4), repeated with NumPy and scikit-learn.
import numpy as np
from sklearn.tree import DecisionTreeClassifier as DTC
gini = lambda c: 1 - ((np.array(c) / sum(c)) ** 2).sum() if sum(c) else 0.0

print("== 25A.1  bootstrap sample of n = 8")
n = 8; draw = [2, 5, 5, 1, 7, 2, 2, 8]; cnt = np.bincount(draw, minlength=n + 1)[1:]
print("counts c_1..c_8 =", cnt.tolist(), " sum", cnt.sum())
print("out of bag:", [i + 1 for i in range(n) if cnt[i] == 0], " distinct points:", int((cnt > 0).sum()))
print("(7/8)^2 = %.6f, ^4 = %.6f, ^8 = %.6f" % ((7 / 8) ** 2, (7 / 8) ** 4, (7 / 8) ** 8))
print("fraction out of bag in this sample: %d/%d = %.4f" % ((cnt == 0).sum(), n, (cnt == 0).mean()))
print("expected number of distinct points: 8 (1 - %.4f) = %.4f; limit n (1 - 1/e) = %.4f" % ((7 / 8) ** 8, 8 * (1 - (7 / 8) ** 8), 8 * (1 - np.exp(-1))))
rng = np.random.default_rng(0); sim = np.mean([len(np.unique(rng.integers(0, 8, 8))) for _ in range(100000)])
print("simulation of 100000 samples: mean distinct %.3f" % sim)

print("\n== 25A.2  variance of an average, sigma^2 = 9, rho = 0.4")
s2, rho = 9.0, 0.4; var = lambda B: rho * s2 + (1 - rho) * s2 / B
print("rho sigma^2 = %.2f, (1 - rho) sigma^2 = %.2f" % (rho * s2, (1 - rho) * s2))
for B in (1, 10, 14, 15, 16, 100): print("B = %3d: variance %.4f   ratio to the limit %.4f" % (B, var(B), var(B) / (rho * s2)))
print("limit x 1.10 = %.4f;  smallest B with var <= that: %d (5.4/B <= 0.36)" % (1.1 * rho * s2, next(B for B in range(1, 100) if var(B) <= 1.1 * rho * s2 + 1e-12)))
C = np.full((10, 10), rho * s2); np.fill_diagonal(C, s2); print("check with the covariance matrix, B = 10: sum of entries / 100 = %.4f" % (C.sum() / 100))

print("\n== 25A.3  soft and hard voting")
P3 = np.array([[0.9, 0.4, 0.45], [0.6, 0.6, 0.1]])
for k, r in enumerate(P3, 1):
    v = (r > 0.5).astype(int); print("sample %d: P = %s  mean %.4f -> class %d;  votes %s -> %d for class 1 of 3 -> class %d" % (k, r.tolist(), r.mean(), int(r.mean() > 0.5), v.tolist(), v.sum(), int(v.sum() >= 2)))
print("sum of P(class 0), P(class 1) per tree is 1; the mean of class 0 for sample 1: %.4f" % (1 - P3[0].mean()))

print("\n== 25A.4  random thresholds, x = 2 3 7 9, y = 0 0 1 1")
x = np.array([2, 3, 7, 9.]); y = np.array([0, 0, 1, 1]); G0 = gini(np.bincount(y))
def dec(t):
    L, R = y[x <= t], y[x > t]; w = (len(L) * gini(np.bincount(L, minlength=2)) + len(R) * gini(np.bincount(R, minlength=2))) / 4
    return w, G0 - w, np.bincount(L, minlength=2), np.bincount(R, minlength=2)
print("root Gini %.4f" % G0)
for t in (2.5, 4.0, 8.0, 5.0):
    w, dd, cl, cr = dec(t); print("threshold %.1f: left (%d,%d) right (%d,%d) Gini %.4f %.4f  weighted %.4f  decrease %.4f" % (t, cl[0], cl[1], cr[0], cr[1], gini(cl), gini(cr), w, dd))
print("exhaustive search: midpoints 2.5, 5.0, 8.0 -> decreases", [round(float(dec(t)[1]), 4) for t in (2.5, 5.0, 8.0)])
st = DTC(max_depth=1, random_state=0).fit(x[:, None], y); print("scikit-learn stump threshold", st.tree_.threshold[0])
print("probability a uniform draw on (2, 9) lies in (3, 7): %d/%d = %.4f" % (7 - 3, 9 - 2, 4 / 7))
rng = np.random.default_rng(1); u = rng.uniform(2, 9, 200000); print("simulation: %.4f" % ((u > 3) & (u < 7)).mean())
print("a draw in (3, 7) gives Delta = 0.5; draws in (2, 3) or (7, 9) give: ", round(dec(2.5)[1], 4), round(dec(8.0)[1], 4))
print("expected decrease of one random threshold: %.4f" % ((4 / 7) * 0.5 + (1 / 7) * dec(2.5)[1] + (2 / 7) * dec(8.0)[1]))
