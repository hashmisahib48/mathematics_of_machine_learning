# Hand checks for Problems 12A.1 - 12A.3 (every number quoted in the solutions)
import numpy as np
from sklearn.linear_model import Ridge
np.set_printoptions(precision=4, suppress=True)

print("== 12A.1  x = -2, -1, 1, 2; y = -3, -1, 2, 2; alpha = 2, lambda = 4")
x = np.array([-2., -1, 1, 2]); y = np.array([-3., -1, 2, 2]); a, lam = 2.0, 4.0
Sxx, Sxy = x @ x, x @ y
print("terms of Sxx:", x ** 2, " Sxx =", Sxx, "  terms of Sxy:", x * y, " Sxy =", Sxy)
prec = lam + a * Sxx; m = a * Sxy / prec
print("precision = lambda + alpha Sxx =", prec, " variance =", 1 / prec, " sd =", prec ** -0.5)
print("mean = alpha Sxy / precision =", a * Sxy, "/", prec, "=", m, "  as a fraction 13/12 =", 13 / 12)
print("least squares:", Sxy / Sxx, "  shrinkage factor Sxx/(Sxx + lambda/alpha) =", Sxx / (Sxx + lam / a), "  m/ols =", m / (Sxy / Sxx))
print("prior sd 1/sqrt(lambda) =", lam ** -0.5, "  ratio posterior sd/prior sd =", prec ** -0.5 / lam ** -0.5)
for r in (lam / a, 4.0, 1.0):
    print(f"  Ridge(alpha={r}, fit_intercept=False) slope =", round(float(Ridge(alpha=r, fit_intercept=False).fit(x[:, None], y).coef_[0]), 4))
print("ridge by hand: Sxy/(Sxx + alpha_ridge) =", Sxy / (Sxx + lam / a))

print("\n== 12A.2  calibration data, alpha = 1, lambda = 5")
xc = np.array([-3., -1, 1, 3]); yc = np.array([-2., -2, 1, 3]); a, lam = 1.0, 5.0
Sxx, Sxy = xc @ xc, xc @ yc; prec = lam + a * Sxx
print("Sxx =", Sxx, " Sxy =", Sxy, " precision =", prec, "  variance of the prediction = 1/alpha + x*^2/precision = 1 + x*^2/%d" % prec)
print("sd = 2 (twice the noise sd 1): 1 + x^2/25 = 4 -> x^2 =", 3 * prec, " x =", np.sqrt(3 * prec))
print("variance at x = 3:", 1 + 9 / prec, " sd =", np.sqrt(1 + 9 / prec), "  weight part =", 9 / prec, "  ratio to the floor =", np.sqrt(1 + 9 / prec))
print("slip: sd = 2 read as variance = 2:  x =", np.sqrt(1 * prec), "; and the sd at x = 8.66:", np.sqrt(1 + 75 / prec))
for xs in (0, 3, 6, 8.66, 10, 20): print(f"  x* = {xs:5.2f}: sd = {np.sqrt(1 + xs ** 2 / prec):.4f}")

print("\n== 12A.3  one update from alpha = 2, lambda = 1 on the calibration data")
a, lam = 2.0, 1.0
prec = lam + a * Sxx; m = a * Sxy / prec; gamma = a * Sxx / prec
print("precision =", prec, " m = 36/41 =", m, " gamma = 40/41 =", gamma)
fit = m * xc; r = yc - fit
print("fitted values:", fit, " residuals:", r, " squares:", r ** 2, " SSE =", (r ** 2).sum())
print("m^2 =", m ** 2, " new lambda = gamma/m^2 =", gamma / m ** 2)
sse = (r ** 2).sum()
print("new alpha = (n - gamma)/SSE = (4 -", round(gamma, 4), ")/", round(sse, 4), "=", (4 - gamma) / sse)
print("converged values (Listing 12.1): alpha 1.667, lambda 1.282.  start (2, 1) -> (%.4f, %.4f)" % ((4 - gamma) / sse, gamma / m ** 2))
print("slips: alpha = n/SSE =", 4 / sse, "; lambda = 1/m^2 (gamma forgotten) =", 1 / m ** 2, "; lambda = p/m^2 with p = 1 is the same")
# iterate to the fixed point from both starts
for a0, l0 in ((1.0, 5.0), (2.0, 1.0)):
    aa, ll = a0, l0
    for k in range(60):
        pr = ll + aa * Sxx; mm = aa * Sxy / pr; g = aa * Sxx / pr; s = ((yc - mm * xc) ** 2).sum(); ll, aa = g / mm ** 2, (4 - g) / s
    print(f"fixed point from ({a0}, {l0}): alpha = {aa:.4f}, lambda = {ll:.4f}, slope = {Sxy*aa/(ll+aa*Sxx):.4f}")
