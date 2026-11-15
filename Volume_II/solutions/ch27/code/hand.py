# Hand checks for Problems 27A.1 - 27A.3.  Run: python3 code/hand.py
import numpy as np
from scipy.special import expit
from sklearn.ensemble import GradientBoostingRegressor as GR

def best_stump(x, r):
    """all thresholds: left mean, right mean, SSE of the two leaves; returns the best one"""
    rows = []
    for t in (x[:-1] + x[1:]) / 2:
        L, R = r[x <= t], r[x > t]
        rows.append((t, L.mean(), R.mean(), ((L - L.mean()) ** 2).sum() + ((R - R.mean()) ** 2).sum()))
    return rows, min(rows, key=lambda q: q[3])

print("== 27A.1  x = 1..4, y = 2, 3, 9, 12, squared error, eta = 1")
x = np.array([1, 2, 3, 4.]); y = np.array([2, 3, 9, 12.]); F = np.full(4, y.mean())
print("F0 =", F[0], " residuals", (y - F).tolist(), " MSE", np.mean((y - F) ** 2), " SST/n check:", np.var(y))
for m in (1, 2):
    r = y - F; rows, b = best_stump(x, r)
    print(f"-- stage {m}: residuals {r.tolist()}")
    for t, lm, rm, sse in rows: print(f"   threshold {t}: left mean {lm:+.4f}  right mean {rm:+.4f}  SSE after {sse:.4f}  (SSE of r: {(r**2).sum():.4f}, gain {(r**2).sum()-sse:.4f})")
    F = F + np.where(x <= b[0], b[1], b[2])
    print(f"   best {b[0]}: leaves {b[1]:+.4f}, {b[2]:+.4f}; F{m} = {F.tolist()}  MSE = {np.mean((y - F) ** 2):.4f}")
g = GR(n_estimators=2, max_depth=1, learning_rate=1.0, random_state=0).fit(x[:, None], y)
print("scikit-learn train_score_ (MSE):", g.train_score_.round(4).tolist())

print("\n== 27A.2  x = 1..6, y = 1, 2, 4, 5, 6, 50, one stump, eta = 1")
x = np.arange(1, 7.); y = np.array([1, 2, 4, 5, 6, 50.])
F0 = y.mean(); r = y - F0; print(f"squared error: F0 = {F0:.4f}, residuals {r.round(4).tolist()}")
rows, b = best_stump(x, r)
for t, lm, rm, sse in rows: print(f"   threshold {t}: left {lm:+.4f} right {rm:+.4f}  SSE {sse:.2f}")
print(f"   best threshold {b[0]}: leaves {b[1]:.4f}, {b[2]:.4f}; predictions {(F0 + np.where(x <= b[0], b[1], b[2])).round(4).tolist()}")
med = np.median(y); r = y - med; s = np.sign(r)
print(f"absolute error: F0 = median = {med}, residuals {r.tolist()}, signs {s.tolist()}")
rows, b = best_stump(x, s)
for t, lm, rm, sse in rows: print(f"   threshold {t}: left mean sign {lm:+.2f} right {rm:+.2f}  SSE of the signs {sse:.3f}")
t = b[0]; ml, mr = np.median(r[x <= t]), np.median(r[x > t])
print(f"   best threshold {t}; medians of the residuals: {ml}, {mr}; predictions {(med + np.where(x <= t, ml, mr)).tolist()}")
print("   with the leaf means of the signs instead (-1, +1) the step would be +-1 and the predictions", (med + np.where(x <= t, -1, 1)).tolist())
print("   mean absolute error: start", np.abs(y - med).mean(), " after the stump", np.abs(y - (med + np.where(x <= t, ml, mr))).mean())
print("   mean squared error: squared-error stump", np.mean((y - (F0 + np.where(x <= 5.5, -7.7333, 38.6667))) ** 2).round(4), "; absolute-error stump", np.mean((y - (med + np.where(x <= t, ml, mr))) ** 2).round(4))
y2 = y.copy(); y2[5] = 6.0
print("   if the last target were 6 instead of 50: mean", y2.mean(), " median", np.median(y2))

print("\n== 27A.3  a Newton leaf, labels 1, 1, 0, probabilities 0.2, 0.5, 0.8")
yl = np.array([1, 1, 0.]); p = np.array([0.2, 0.5, 0.8]); r = yl - p
print("r =", r.tolist(), " sum r =", r.sum().round(4), " p(1-p) =", (p * (1 - p)).round(4).tolist(), " sum =", (p * (1 - p)).sum().round(4))
gam = r.sum() / (p * (1 - p)).sum(); print(f"gamma = {gam:.4f}   plain average of r = {r.mean():.4f}")
F1 = np.log(p[0] / (1 - p[0])); print(f"score of sample 1: {F1:.4f} -> {F1 + 0.5*gam:.4f}; probability {p[0]:.4f} -> {expit(F1 + 0.5*gam):.4f}")
print("new scores of the three samples:", (np.log(p / (1 - p)) + 0.5 * gam).round(4).tolist(), " probabilities", expit(np.log(p / (1 - p)) + 0.5 * gam).round(4).tolist())
lg = lambda g: -(yl * np.log(expit(np.log(p / (1 - p)) + g)) + (1 - yl) * np.log(1 - expit(np.log(p / (1 - p)) + g))).sum()
gs = np.linspace(-2, 4, 60001); print(f"exact minimiser of the leaf loss: {gs[np.argmin([lg(g) for g in gs])]:.4f}; loss at 0: {lg(0):.4f}, at the Newton step: {lg(gam):.4f}, at the plain average: {lg(r.mean()):.4f}")

print("\n== 27B.3  side remark: the test curve of one fit without early stopping, Friedman data, eta = 0.1, depth 3")
from sklearn.datasets import make_friedman1
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
Xf, yf = make_friedman1(n_samples=600, noise=1.0, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(Xf, yf, test_size=0.5, random_state=0)
g = GR(n_estimators=1000, learning_rate=0.1, random_state=0).fit(Xtr, ytr)
te = np.array([mean_squared_error(yte, p) for p in g.staged_predict(Xte)])
print("test MSE after 78, 102, 173, 300, 500, 1000 trees:", [round(float(te[k - 1]), 3) for k in (78, 102, 173, 300, 500, 1000)], " best", te.min().round(3), "at", int(te.argmin()) + 1)
print("test MSE between 78 and 173 trees: min %.3f max %.3f" % (te[77:173].min(), te[77:173].max()))
