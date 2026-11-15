# Problem 23C.1: which of the five points can be moved by 0.2 without changing the fit?
import numpy as np
from sklearn.svm import SVR

x = np.array([1., 2., 3., 4., 5.]); y = np.array([2.0, 2.4, 3.6, 3.8, 5.9]); X = x[:, None]
fit = lambda yy: SVR(kernel="linear", C=10, epsilon=0.5, tol=1e-12).fit(X, yy)
m0 = fit(y); w0, b0 = m0.coef_[0, 0], m0.intercept_[0]
r = y - m0.predict(X); beta = np.zeros(5); beta[m0.support_] = m0.dual_coef_[0]
print("fit: w %.4f b %.4f" % (w0, b0))
print("residuals r:", r.round(3))
print("beta       :", beta.round(3), "  (C = 10)")
kind = ["inside" if abs(v) < 0.5 - 1e-9 else "edge (free)" if abs(abs(v) - 0.5) < 1e-9 else "outside (bound)" for v in r]
print("kind       :", kind)
print("point  move   new residual (old line)   w        b        changed")
for k, name in enumerate("ABCDE"):
    for d in (+0.2, -0.2):
        yy = y.copy(); yy[k] += d; m = fit(yy)
        ch = abs(m.coef_[0, 0] - w0) > 1e-7 or abs(m.intercept_[0] - b0) > 1e-7
        print("  %s   %+.1f   %+.3f                   %.4f   %.4f   %s" % (name, d, r[k] + d, m.coef_[0, 0], m.intercept_[0], "yes" if ch else "no"))
print("point D moved up by t: first change at |r| = 0.5, i.e. t = 0.125; scan")
for t in (0.10, 0.125, 0.13, 0.15):
    yy = y.copy(); yy[3] += t; m = fit(yy); print("  t = %.3f  new residual %.3f  w %.4f  b %.4f" % (t, r[3] + t, m.coef_[0, 0], m.intercept_[0]))
yy = y.copy(); yy[3] -= 5.0
m = fit(yy); print("point D moved down by 5: w %.4f  b %.4f" % (m.coef_[0, 0], m.intercept_[0]))
