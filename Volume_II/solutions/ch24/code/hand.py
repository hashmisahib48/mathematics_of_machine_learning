# Hand checks for the Chapter 24 solutions (24A.1 - 24A.4), repeated with scikit-learn.
import numpy as np
from sklearn.tree import DecisionTreeClassifier as DTC, DecisionTreeRegressor as DTR
np.set_printoptions(precision=4, suppress=True)
gini = lambda c: 1 - ((np.array(c) / sum(c)) ** 2).sum()
def ent(c):
    p = np.array(c, float) / sum(c); p = p[p > 0]; return -(p * np.log2(p)).sum()

print("== 24A.1  node (6, 4, 2)")
c = [6, 4, 2]; p = np.array(c) / 12
print("p =", p.round(4).tolist(), " p^2 =", (p ** 2).round(4).tolist(), " sum p^2 = %.4f" % (p ** 2).sum())
print("Gini = %.4f   misclassification = %.4f" % (gini(c), 1 - p.max()))
print("-p log2 p =", (-p * np.log2(p)).round(4).tolist(), "  entropy = %.4f" % ent(c))
print("fractions: 1/2, 1/3, 1/6;  1 - 1/4 - 1/9 - 1/36 = %.4f = 22/36 = %.4f" % (1 - 1/4 - 1/9 - 1/36, 22/36))
print("maximum for K = 3: Gini %.4f entropy %.4f" % (gini([1, 1, 1]), ent([1, 1, 1])))

print("\n== 24A.2  x = 1..8, classes 0 0 1 0 0 1 1 1")
x = np.arange(1, 9); y = np.array([0, 0, 1, 0, 0, 1, 1, 1]); G0 = gini(np.bincount(y))
print("root counts", np.bincount(y).tolist(), " Gini %.4f" % G0)
print("thr  left(0/1) right(0/1)  G_L    G_R    weighted")
tab = []
for t in x[:-1] + 0.5:
    L, R = y[x <= t], y[x > t]; cl, cr = np.bincount(L, minlength=2), np.bincount(R, minlength=2)
    w = (len(L) * gini(cl) + len(R) * gini(cr)) / 8; tab.append(w)
    print("%.1f   %d/%d      %d/%d       %.4f %.4f %.4f" % (t, cl[0], cl[1], cr[0], cr[1], gini(cl), gini(cr), w))
k = int(np.argmin(tab)); print("best threshold %.1f weighted %.4f decrease %.4f" % (x[k] + 0.5, tab[k], G0 - tab[k]))
print("right part of best split: 3 of 3 pure? classes", y[x > 5.5].tolist(), " left", y[x <= 5.5].tolist())
t = DTC(max_depth=1, random_state=0).fit(x[:, None], y); print("sklearn stump threshold", t.tree_.threshold[0], " impurities", t.tree_.impurity[:3].round(4).tolist())
print("decrease weighted by N_t/N (root, N_t = N): ", round(G0 - tab[k], 4))

print("\n== 24A.3  x = 1..5, y = 2 3 10 11 12")
x = np.arange(1, 6.); y = np.array([2, 3, 10, 11, 12.]); sse = lambda v: ((v - v.mean()) ** 2).sum()
print("mean %.1f  SST = %.2f  impurity %.3f" % (y.mean(), sse(y), sse(y) / 5))
print("thr  mean_L mean_R  SSE_L  SSE_R  total")
for t in x[:-1] + 0.5:
    L, R = y[x <= t], y[x > t]; print("%.1f  %.4f %.4f  %.4f %.4f %.4f" % (t, L.mean(), R.mean(), sse(L), sse(R), sse(L) + sse(R)))
L, R = y[x <= 2.5], y[x > 2.5]; S = sse(L) + sse(R)
print("deviations left", (L - L.mean()).tolist(), "right", (R - R.mean()).tolist())
print("best SSE %.2f  decrease %.2f  R2 = 1 - %.2f/%.2f = %.4f" % (S, sse(y) - S, S, sse(y), 1 - S / sse(y)))
m = DTR(max_depth=1, random_state=0).fit(x[:, None], y); print("sklearn threshold", m.tree_.threshold[0], "values", m.tree_.value[1:, 0, 0].tolist(), " R2", round(m.score(x[:, None], y), 4))
print("impurity decrease per sample (Eq. 24.4-2): %.4f -> %.4f" % (sse(y) / 5, S / 5))

print("\n== 24A.4  weakest link")
R = {0: .20, 1: .08, 2: .04, 3: .02, 4: .01}
a1 = (R[1] - (R[3] + R[4])) / (2 - 1); a0 = (R[0] - (R[2] + R[3] + R[4])) / (3 - 1)
print("leaves under t1: t3, t4 (sum %.2f) -> alpha(t1) = (%.2f - %.2f)/1 = %.4f" % (R[3] + R[4], R[1], R[3] + R[4], a1))
print("leaves under t0: t2, t3, t4 (sum %.2f) -> alpha(t0) = (%.2f - %.2f)/2 = %.4f" % (R[2] + R[3] + R[4], R[0], R[2] + R[3] + R[4], a0))
a0b = (R[0] - (R[1] + R[2])) / (2 - 1); print("after cutting t1: leaves t1, t2 (sum %.2f) -> alpha(t0) = %.4f" % (R[1] + R[2], a0b))
for al in (0.06,):
    for name, Rt, nl in (("full tree", R[2] + R[3] + R[4], 3), ("t1 cut", R[1] + R[2], 2), ("root only", R[0], 1)):
        print("alpha = %.2f  %-9s R = %.2f, leaves %d, R_alpha = %.3f" % (al, name, Rt, nl, Rt + al * nl))
print("sequence of alphas 0, %.3f, %.3f, %.3f" % (a1, a0b, a0b))
print("the intervals: full tree for alpha < %.3f, two leaves for %.3f <= alpha < %.3f, root for alpha >= %.3f" % (a1, a1, a0b, a0b))
