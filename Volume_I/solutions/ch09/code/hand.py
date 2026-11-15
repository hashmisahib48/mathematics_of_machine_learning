# Hand checks for Problems 9A.1 - 9A.4, 9B-support and 9C.1 (every number quoted in the solutions)
import numpy as np
from sklearn.linear_model import LinearRegression
np.set_printoptions(precision=4, suppress=True)

print("== 9A.1  (0,1) (1,3) (2,4) (3,6)")
x = np.array([0, 1, 2, 3.]); y = np.array([1, 3, 4, 6.])
xb, yb = x.mean(), y.mean(); print("means:", xb, yb)
print("x-xb:", x - xb, " y-yb:", y - yb, " products:", (x - xb) * (y - yb), " squares:", (x - xb) ** 2)
Sxy = ((x - xb) * (y - yb)).sum(); Sxx = ((x - xb) ** 2).sum(); print("Sxy =", Sxy, " Sxx =", Sxx)
w = Sxy / Sxx; b = yb - w * xb; print("w =", w, " b =", b)
yh = b + w * x; r = y - yh; print("yhat:", yh, " r:", r, " r^2:", r ** 2, " sum r:", r.sum().round(12), " SSE =", (r ** 2).sum().round(6))
m = LinearRegression().fit(x[:, None], y); print("sklearn:", m.coef_[0], m.intercept_)
print("sum x*r =", (x * r).sum().round(12))

print("\n== 9A.2  rival line 1 + 1.6x on the deliveries")
xd = np.array([1, 2, 3, 4, 5.]); yd = np.array([3, 4, 6, 7, 10.])
p = 1 + 1.6 * xd; r2 = yd - p
print("yhat:", p, " r:", r2, " r^2:", r2 ** 2, " SSE =", (r2 ** 2).sum().round(6), " sum r =", r2.sum().round(6))
ls = 0.9 + 1.7 * xd; sse0 = ((yd - ls) ** 2).sum(); print("least-squares SSE =", sse0.round(6))
db, dw = 1 - 0.9, 1.6 - 1.7
print("(db + dw x):", db + dw * xd, " squares:", (db + dw * xd) ** 2, " sum =", ((db + dw * xd) ** 2).sum().round(6))
print("SSE_rival - SSE_ls =", ((r2 ** 2).sum() - sse0).round(6))

print("\n== 9A.3  line through the origin")
Sxy0, Sxx0, Syy0 = (xd * yd).sum(), (xd * xd).sum(), (yd * yd).sum(); print("sum xy, sum x^2, sum y^2:", Sxy0, Sxx0, Syy0)
w0 = Sxy0 / Sxx0; print("w =", w0)
p0 = w0 * xd; r0 = yd - p0; print("yhat:", p0, " r:", r0, " r^2:", r0 ** 2, " SSE =", (r0 ** 2).sum().round(6), " sum r =", r0.sum().round(6))
print("SSE from the shortcut  sum y^2 - (sum xy)^2/sum x^2 =", (Syy0 - Sxy0 ** 2 / Sxx0).round(6))
print("sum x*r =", (xd * r0).sum().round(10))
db, dw = 0.9 - 0.0, 1.7 - w0
print("excess over 1.10: db =", db, " dw =", round(dw, 6), " sum (db+dw x)^2 =", ((db + dw * xd) ** 2).sum().round(6), " 1.836-1.10 =", ((r0 ** 2).sum() - sse0).round(6))
print("sklearn fit_intercept=False:", LinearRegression(fit_intercept=False).fit(xd[:, None], yd).coef_[0])

print("\n== 9A.4  R^2 for the four samples")
sst = ((y - yb) ** 2).sum(); sse = (r ** 2).sum(); print("SST =", sst, " SSE =", sse.round(6), " R2 =", (1 - sse / sst).round(6))
print("SSR = SST - SSE =", (sst - sse).round(6), "  w^2 Sxx =", (w ** 2 * Sxx).round(6), "  Sxy^2/(Sxx Syy) =", (Sxy ** 2 / (Sxx * sst)).round(6), " 64/65 =", round(64 / 65, 6))

print("\n== 9C.1  fit_intercept=False")
print("sum r =", r0.sum().round(6), "   n (ybar - w xbar) =", (5 * (yd.mean() - w0 * xd.mean())).round(6), "  w*xbar =", round(w0 * 3, 4))
print("mean residual =", r0.mean().round(4), "  with intercept sum r =", (yd - ls).sum().round(12))

print("\n== common slip in 9A.1: dividing the raw sums (the through-origin formula)")
print("sum xy =", (x * y).sum(), " sum x^2 =", (x * x).sum(), " ratio =", ((x * y).sum() / (x * x).sum()).round(4),
      "  Sxy from raw sums = sum xy - n xbar ybar =", (x * y).sum() - 4 * xb * yb)
