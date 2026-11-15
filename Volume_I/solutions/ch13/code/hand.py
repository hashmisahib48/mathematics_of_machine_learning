# Hand checks for Problems 13A.1 - 13A.4 (every number quoted in the solutions)
import numpy as np
from itertools import combinations
np.set_printoptions(precision=4, suppress=True)

print("== 13A.1  eps = 1.35, sigma = 2, residuals -1, 2, 3, -6, 12")
eps, sg = 1.35, 2.0; r = np.array([-1., 2, 3, -6, 12])
t = eps * sg; print("threshold eps*sigma =", t)
print("|r|        :", abs(r)); print("|r|/sigma  :", abs(r) / sg)
om = np.minimum(1, t / abs(r)); print("omega      :", om, " (eps*sigma/|r| =", t / abs(r), ")")
print("omega*r    :", om * r, " flagged (|r| > eps*sigma):", abs(r) > t)

print("\n== 13A.2  f = 0.6, gamma = 0.99")
f = 0.6
for s, g in ((3, 0.99), (2, 0.99), (3, 0.999)):
    fs = f ** s; fail = 1 - fs; num = np.log(1 - g); den = np.log(fail)
    print("s=%d gamma=%.3f: f^s=%.3f 1-f^s=%.3f ln(1-gamma)=%.4f ln(1-f^s)=%.4f ratio=%.3f -> N=%d" % (s, g, fs, fail, num, den, num / den, int(np.ceil(num / den))))
N = 19; print("check s=3: (1-f^s)^N for N=18, 19:", (1 - f ** 3) ** 18, (1 - f ** 3) ** 19, " limit 0.01")
print("for comparison, f = 0.6 and s = 11 (the 10-feature model):", np.ceil(np.log(0.01) / np.log(1 - f ** 11)))

print("\n== 13A.3  Theil-Sen")
x = np.arange(5.); y = np.array([1., 3, 5, 7, 30])
sl = []
for i, j in combinations(range(5), 2):
    s = (y[j] - y[i]) / (x[j] - x[i]); sl.append(s); print("points %d,%d: (%g,%g)-(%g,%g) slope %.2f" % (i + 1, j + 1, x[i], y[i], x[j], y[j], s))
ss = np.sort(sl); print("sorted:", ss, " median:", np.median(sl))
w = np.median(sl); ic = y - w * x; print("intercepts y - w x:", ic, " median:", np.median(ic))
b = np.median(ic)
xb, yb = x.mean(), y.mean(); Sxy = ((x - xb) * (y - yb)).sum(); Sxx = ((x - xb) ** 2).sum()
print("OLS: xbar %.1f ybar %.1f Sxy %.1f Sxx %.1f slope %.2f intercept %.2f" % (xb, yb, Sxy, Sxx, Sxy / Sxx, yb - Sxy / Sxx * xb))
print("predictions at x = 5: Theil-Sen %.1f  OLS %.1f ; true line 1+2x gives %.1f" % (b + w * 5, yb - Sxy / Sxx * xb + Sxy / Sxx * 5, 1 + 2 * 5))
print("residuals of the Theil-Sen line:", y - b - w * x, "; of OLS:", y - (yb - Sxy / Sxx * xb) - Sxy / Sxx * x)
cl = [(y[j] - y[i]) / (x[j] - x[i]) for i, j in combinations(range(4), 2)]; print("slopes among the four good points:", cl)
print("pairs free of point 5:", len(cl), "of", len(sl))

print("\n== 13A.4  tau = 0.25, targets 1, 2, 3, 4, 10")
d = np.array([1., 2, 3, 4, 10]); tau = 0.25
for c in (2, 3):
    rr = d - c; loss = np.where(rr >= 0, tau * rr, (tau - 1) * rr)
    print("c=%d residuals y-c:" % c, rr, " losses:", loss, " total %.2f" % loss.sum())
for c in (1, 2, 3, 4, 10):
    rr = d - c; print("c=%2d total %.2f" % (c, np.where(rr >= 0, tau * rr, (tau - 1) * rr).sum()))
# rate of change, Eq. 13.5-2: (1-tau) N_below - tau N_above, just beside c
for c in (2, 3):
    for side, eps_ in (("just below", -1e-6), ("just above", 1e-6)):
        cc = c + eps_; nb = (d < cc).sum(); na = (d > cc).sum(); print("c=%d %s: N_below %d N_above %d rate %.2f" % (c, side, nb, na, (1 - tau) * nb - tau * na))
print("share of the targets that are <= 2: %d/5 = %.2f ; <= 1: 1/5 = 0.20" % ((d <= 2).sum(), (d <= 2).mean()))

print("\n== extra checks used in the notes")
print("13A.2: 0.64^10 = %.4f, 0.64^11 = %.4f; 0.784^28 = %.4f, 0.784^29 = %.4f; ln(10)/0.2433 = %.2f" % (0.64 ** 10, 0.64 ** 11, 0.784 ** 28, 0.784 ** 29, np.log(10) / -np.log(0.784)))
print("13A.3: mean of the ten slopes = %.3f; fraction of pairs free of the wrong point = %d/10" % (np.mean(sl), 6))
import warnings; warnings.simplefilter('ignore')
from sklearn.linear_model import TheilSenRegressor
ts = TheilSenRegressor(random_state=0).fit(x[:, None], y); print("13A.3: TheilSenRegressor (spatial median of the 10 pairs): intercept %.3f slope %.3f, predicts %.3f at x = 5" % (ts.intercept_, ts.coef_[0], ts.predict([[5.]])[0]))
print("13A.1: comparing |r| with eps instead of eps*sigma would flag", np.where(abs(r) > eps)[0] + 1)
