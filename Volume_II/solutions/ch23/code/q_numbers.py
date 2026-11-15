# Numbers quoted in the model answers to the questions of Chapter 23.
import numpy as np
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor as TTR
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X = np.array([[1.], [2.], [3.], [4.], [5.]]); y = np.array([3., 4., 6., 7., 10.])
fit = lambda yy, C, e: SVR(kernel="linear", C=C, epsilon=e, tol=1e-10).fit(X, yy)
print("== Q1  the delivery times, C = 1, eps = 1")
m = fit(y, 1, 1); print("clean fit: w %.4f b %.4f, support vectors %s" % (m.coef_[0, 0], m.intercept_[0], m.support_ + 1))
for nm, yy, keep in (("sample 1 removed", y, [1, 2, 3, 4]), ("sample 3 removed", y, [0, 1, 3, 4])):
    mm = SVR(kernel="linear", C=1, epsilon=1, tol=1e-10).fit(X[keep], yy[keep]); print("%-20s w %.4f b %.4f" % (nm, mm.coef_[0, 0], mm.intercept_[0]))
for nm, k, new in (("y_3 6 -> 6.5", 2, 6.5), ("y_1 3 -> 3.6", 0, 3.6), ("y_4 7 -> 7.3", 3, 7.3)):
    yy = y.copy(); yy[k] = new; mm = fit(yy, 1, 1); print("%-20s w %.4f b %.4f   (residual %.3f)" % (nm, mm.coef_[0, 0], mm.intercept_[0], new - mm.predict(X[k:k + 1])[0]))
for nm, k, new in (("y_5 10 -> 10.5", 4, 10.5), ("y_2 4 -> 4.3", 1, 4.3)):
    yy = y.copy(); yy[k] = new; mm = fit(yy, 1, 1); print("support vector %-14s w %.4f b %.4f" % (nm, mm.coef_[0, 0], mm.intercept_[0]))
print("wrong entry y_5 = 20, 30, 100 at C = 1:")
for v in (20., 30., 100.):
    yy = y.copy(); yy[4] = v; mm = fit(yy, 1, 1); print("   y_5 = %5.0f: w %.4f b %.4f  beta %s" % (v, mm.coef_[0, 0], mm.intercept_[0], mm.dual_coef_[0].round(3)))
print("== Q2  large epsilon")
for e in (3.4, 3.49, 3.5, 3.6, 5.0):
    mm = fit(y, 10, e); print("eps = %-4g w %.4f b %.4f SV %d" % (e, mm.coef_[0, 0], mm.intercept_[0], len(mm.support_)))
print("half range of the targets:", (y.max() - y.min()) / 2, " midrange:", (y.max() + y.min()) / 2)
rg = np.random.default_rng(1); xa = rg.uniform(0, 6, 300); ya = np.sin(xa) + 0.25 * rg.normal(size=300); Xa = xa[:, None]
for e in (0, 0.05, 0.1, 0.25, 0.5, 0.8):
    mm = SVR(C=10, gamma=0.5, epsilon=e).fit(Xa[:200], ya[:200]); print("sine, noise 0.25, eps = %-4g: %3d support vectors of 200" % (e, len(mm.support_)))
print("== Q3  5 per cent wrong targets, line y = 2x + 1")
rg = np.random.default_rng(0); n = 100; xs = rg.uniform(0, 10, n); ys = 2 * xs + 1 + 0.5 * rg.normal(size=n); Xl = xs[:, None]
bad = rg.choice(n, 5, replace=False); yw = ys.copy(); yw[bad] += 30
for nm, mdl in (("least squares", LinearRegression()), ("SVR C=1 eps=0.5", SVR(kernel="linear", C=1, epsilon=0.5, tol=1e-8))):
    c0 = mdl.fit(Xl, ys); a = (c0.coef_ if nm == "least squares" else c0.coef_[0]).ravel()[0]; b0 = float(np.ravel(c0.intercept_)[0])
    c1 = mdl.fit(Xl, yw); a1 = (c1.coef_ if nm == "least squares" else c1.coef_[0]).ravel()[0]; b1 = float(np.ravel(c1.intercept_)[0])
    print("%-16s clean: slope %.3f intercept %.3f   5 wrong: slope %.3f intercept %.3f   shift of f at x = 5: %.3f" % (nm, a, b0, a1, b1, (a1 * 5 + b1) - (a * 5 + b0)))
mm = SVR(kernel="linear", C=1, epsilon=0.5, tol=1e-8).fit(Xl, yw); bt = mm.dual_coef_[0]
print("beta of the 5 wrong samples:", np.round(np.sort(bt[np.isin(mm.support_, bad)]), 3), " largest |beta| overall:", np.abs(bt).max().round(3))
print("== Q4  scale of y")
for s in (1, 1000):
    mm = SVR(kernel="linear", C=1, epsilon=0.1, tol=1e-10).fit(X, s * y); print("deliveries x %-4d C = 1, eps = 0.1: w/s %.4f  b/s %.4f" % (s, mm.coef_[0, 0] / s, mm.intercept_[0] / s))
m1 = SVR(kernel="linear", C=1, epsilon=0.1, tol=1e-10).fit(X, y); m2 = SVR(kernel="linear", C=1000, epsilon=100, tol=1e-10).fit(X, 1000 * y)
print("C = 1000, eps = 100 on 1000 y against 1000 x the fit with (1, 0.1): max diff %.1e (in units of 1000 y)" % np.abs(m2.predict(X) / 1000 - m1.predict(X)).max())
Xd, yd = load_diabetes(return_X_y=True); Xtr, Xte, ytr, yte = train_test_split(Xd, yd, test_size=0.25, random_state=0)
print("diabetes, SVR rbf pipeline with a StandardScaler, test R2 (sd of y = %.1f):" % yd.std())
for s, C, e in ((1, 1, 0.1), (1000, 1, 0.1), (1000, 1000, 100)):
    mm = make_pipeline(StandardScaler(), SVR(C=C, epsilon=e)).fit(Xtr, s * ytr); print("   y x %-5d C = %-5g eps = %-5g R2 = %.3f" % (s, C, e, mm.score(Xte, s * yte)))
print("== Q5  why a tube when the noise is Gaussian")
rg = np.random.default_rng(5); n = 200; xs = np.sort(rg.uniform(0, 6, n)); ys = np.sin(xs) + 0.25 * rg.normal(size=n); yw = ys.copy(); bad = rg.choice(n, 10, replace=False); yw[bad] += 3
xt = np.linspace(0, 6, 400)[:, None]
for nm, mdl in (("SVR eps = 0", SVR(C=10, gamma=0.5, epsilon=0)), ("SVR eps = 0.3", SVR(C=10, gamma=0.5, epsilon=0.3)), ("KRR alpha = 0.1", KernelRidge(alpha=0.1, kernel="rbf", gamma=0.5))):
    out = []
    for yy in (ys, yw):
        mm = mdl.fit(xs[:, None], yy - yy.mean()); p = mm.predict(xt) + yy.mean(); out.append(np.sqrt(np.mean((p - np.sin(xt[:, 0]))**2)))
    nsv = len(mdl.support_) if hasattr(mdl, "support_") else n
    print("%-16s RMSE against sin x: clean %.3f, with 10 of 200 targets +3: %.3f   support vectors %d" % (nm, out[0], out[1], nsv))
