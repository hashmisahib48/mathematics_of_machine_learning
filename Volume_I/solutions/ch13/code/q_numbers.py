# Numbers quoted in the model answers to the questions of Chapter 13
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../ch13/code'); os.chdir(HERE + '/..')
import numpy as np
from sklearn.datasets import load_diabetes, make_regression
from sklearn.linear_model import (HuberRegressor, LinearRegression, QuantileRegressor, RANSACRegressor)
from sklearn.model_selection import validation_curve, train_test_split
from sklearn.metrics import mean_absolute_error

x = np.arange(1., 6); y = np.array([3., 4, 6, 7, 20]); X = x[:, None]

print("== Q1  deleting or robust")
ols = LinearRegression().fit(X, y); r = y - ols.predict(X)
print("OLS fit b=%.2f w=%.2f residuals %s" % (ols.intercept_, ols.coef_[0], np.round(r, 2)))
k = np.argmax(abs(r)); keep = np.arange(5) != k
o2 = LinearRegression().fit(X[keep], y[keep])
print("delete the point with the largest |residual| (point %d): slope %.2f intercept %.2f, predicts %.2f at 6 km" % (k + 1, o2.coef_[0], o2.intercept_, o2.predict([[6.]])[0]))
c = LinearRegression().fit(X[:4], y[:4]); print("delete point 5: predicts %.2f at 6 km" % c.predict([[6.]])[0])
h = HuberRegressor().fit(X, y); print("Huber predicts %.2f at 6 km, flags points %s" % (h.predict([[6.]])[0], np.where(h.outliers_)[0] + 1))
# case study
Xd, yd = load_diabetes(return_X_y=True); Xtr, Xte, ytr, yte = train_test_split(Xd, yd, test_size=0.25, random_state=0)
rng = np.random.default_rng(0); bad = rng.choice(len(ytr), size=33, replace=False); yb = ytr.copy(); yb[bad] += 300
hh = HuberRegressor(max_iter=1000).fit(Xtr, yb); fl = hh.outliers_
print("diabetes: Huber flags %d of %d, of which wrecked %d, clean %d (%.1f per cent of the clean records)" % (fl.sum(), len(ytr), fl[bad].sum(), fl.sum() - fl[bad].sum(), 100 * (fl.sum() - fl[bad].sum()) / (len(ytr) - 33)))
keep = ~fl; dl = LinearRegression().fit(Xtr[keep], yb[keep]); print("OLS after deleting the flagged: test MAE %.1f; Huber: %.1f; OLS on all damaged: %.1f"
      % (mean_absolute_error(yte, dl.predict(Xte)), mean_absolute_error(yte, hh.predict(Xte)), mean_absolute_error(yte, LinearRegression().fit(Xtr, yb).predict(Xte))))
# leverage: bad points far out in x
rng = np.random.default_rng(1); xx = rng.uniform(0, 10, 40); yy = 1 + 0.8 * xx + 0.6 * rng.normal(size=40)
xx[:8] = rng.uniform(9, 10, 8); yy[:8] = 1 + 0.8 * xx[:8] - 12 + rng.normal(size=8)
for nm, m in (("OLS", LinearRegression()), ("Huber", HuberRegressor(alpha=0, max_iter=1000)), ("median", QuantileRegressor(alpha=0.0))):
    m.fit(xx[:, None], yy); print("leverage data, %-6s slope %.2f" % (nm, m.coef_[0]))
m = LinearRegression().fit(xx[8:, None], yy[8:]); print("leverage data, bad points deleted: slope %.2f (true 0.8)" % m.coef_[0])

print("\n== Q2  the cliff")
for e in (1.35, 1.40, 1.41, 1.42, 1.45):
    hu = HuberRegressor(epsilon=e, alpha=0, max_iter=5000, tol=1e-12).fit(X, y); print("eps %.2f slope %.3f scale %.3f flagged %d" % (e, hu.coef_[0], hu.scale_, hu.outliers_.sum()))
Xv, yv = make_regression(n_samples=300, n_features=5, n_informative=5, noise=8, random_state=1)
rng = np.random.default_rng(3); Xt, yt = Xv[200:], yv[200:]; Xv, yv = Xv[:200], yv[:200].copy()
bd = rng.choice(200, 24, replace=False); yv[bd] += rng.choice([-1, 1], 24) * rng.uniform(150, 300, 24)
eps = np.round(np.logspace(0, np.log10(8), 15), 3)
_, va = validation_curve(HuberRegressor(alpha=0, max_iter=2000), Xv, yv, param_name='epsilon', param_range=eps, cv=5, scoring='neg_median_absolute_error')
cv = -va.mean(1); tst = [mean_absolute_error(yt, HuberRegressor(epsilon=e, alpha=0, max_iter=2000).fit(Xv, yv).predict(Xt)) for e in eps]
for e, a, b in zip(eps, cv, tst): print("eps %.3f  CV median error %.2f  test MAE %.2f" % (e, a, b))
pl = eps[cv <= 1.05 * cv.min()]; print("eps within 5 per cent of the best CV error:", pl.min(), "to", pl.max(), " best", eps[cv.argmin()])

print("\n== Q3  sensitivity of the median fit")
def lad(yy): q = QuantileRegressor(alpha=0.0).fit(X, yy); return q.intercept_, q.coef_[0]
def fit(yy): o = LinearRegression().fit(X, yy); return o.intercept_, o.coef_[0]
print("LAD  b=%.3f w=%.3f   OLS b=%.3f w=%.3f" % (*lad(y), *fit(y)))
print("line values at the points:", np.round(lad(y)[0] + lad(y)[1] * x, 2), " passes through points", np.where(abs(y - lad(y)[0] - lad(y)[1] * x) < 1e-9)[0] + 1)
for lab, yy in (("y5 = 20 -> 200", [3, 4, 6, 7, 200]), ("y5 = 20 -> 8", [3, 4, 6, 7, 8]), ("y2 = 4 -> 4.3 (stays below the line)", [3, 4.3, 6, 7, 20]), ("y2 = 4 -> 5 (crosses the line)", [3, 5, 6, 7, 20]), ("y3 = 6 -> 6.3 (on the line)", [3, 4, 6.3, 7, 20])):
    yy = np.array(yy, float); print("%-40s LAD (%.3f, %.3f)   OLS (%.3f, %.3f)" % (lab, *lad(yy), *fit(yy)))
print("OLS slope change per unit change of y_i: (x_i - xbar)/Sxx =", np.round((x - x.mean()) / ((x - x.mean()) ** 2).sum(), 2))

print("\n== Q4  scale of the threshold")
for unit, f in (("minutes", 1.0), ("seconds", 60.0)):
    hu = HuberRegressor(alpha=0, max_iter=2000, tol=1e-10).fit(X, y * f); ra = RANSACRegressor(residual_threshold=2.0, random_state=0).fit(X, y * f)
    rd = RANSACRegressor(random_state=0).fit(X, y * f)
    print("%-8s Huber(1.35) slope/f = %.3f flagged %s | RANSAC t=2 inliers %s slope/f = %.3f | RANSAC default slope/f = %.3f"
          % (unit, hu.coef_[0] / f, np.where(hu.outliers_)[0] + 1, np.where(ra.inlier_mask_)[0] + 1, ra.estimator_.coef_[0] / f, rd.estimator_.coef_[0] / f))
mad = np.median(abs(y - np.median(y))); print("default RANSAC threshold (MAD of y) = %.1f min" % mad)
