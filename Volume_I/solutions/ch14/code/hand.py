# Hand checks for Problems 14A.1 - 14A.3 (every number quoted in the solutions)
import numpy as np
np.set_printoptions(precision=4, suppress=True)
dP = lambda y, m: 2 * (np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / m), 0) - (y - m))      # Eq. 14.3-6
dG = lambda y, m: 2 * ((y - m) / m - np.log(y / m))                                              # Eq. 14.3-7

print("== 14A.1  counts 2, 0, 3")
y = np.array([2., 0, 3]); mu = y.mean()
print("sum y = %g, n = %d, mu_hat = %.4f" % (y.sum(), len(y), mu))
ll = lambda m: y.sum() * np.log(m) - len(y) * m
print("l(mu) = 5 ln mu - 3 mu; dl/dmu at 5/3 = %.1e; l(5/3) = %.4f; second derivative -5/mu^2 = %.4f" % (5 / mu - 3, ll(mu), -5 / mu ** 2))
for yi in y:
    if yi > 0: print("y = %g: 2[y ln(y/mu) - (y - mu)] = 2[%.4f - %.4f] = %.4f   (ln(y/mu) = %.4f)" % (yi, yi * np.log(yi / mu), yi - mu, dP(yi, mu), np.log(yi / mu)))
    else: print("y = 0: d = 2 mu = %.4f" % (2 * mu))
d = dP(y, mu); print("unit deviances:", d, " total D = %.4f" % d.sum())
lsat = sum(v * np.log(v) - v for v in y if v > 0); print("check: 2 (l_saturated - l) = 2 (%.4f - %.4f) = %.4f" % (lsat, ll(mu), 2 * (lsat - ll(mu))))
print("deviance at mu = 1.5 and 2 (it is larger): %.4f, %.4f" % (dP(y, 1.5).sum(), dP(y, 2.0).sum()))

print("\n== 14A.2  y = 3, mu = 2 and 4.5")
for m in (2., 4.5):
    print("mu = %.1f: y/mu = %.4f  Poisson 2[3 ln(%.4f) - (3 - %.1f)] = 2[%.4f - %.4f] = %.4f   gamma 2[(3 - %.1f)/%.1f - ln(%.4f)] = 2[%.4f - %.4f] = %.4f"
          % (m, 3 / m, 3 / m, m, 3 * np.log(3 / m), 3 - m, dP(3., m), m, m, 3 / m, (3 - m) / m, np.log(3 / m), dG(3., m)))
print("ratios over / under: Poisson %.4f, gamma %.4f" % (dP(3., 4.5) / dP(3., 2.), dG(3., 2.) / dG(3., 4.5)))
print("squared error: (3-2)^2 = 1 and (3-4.5)^2 = 2.25")

print("\n== 14A.3  x = 0, 1, 2; y = 1, 3, 5; start mu = 3")
x = np.array([0., 1, 2]); y = np.array([1., 3, 5]); X = np.c_[np.ones(3), x]
b0 = np.array([np.log(3), 0.]); mu0 = np.exp(X @ b0); print("start b = ln 3 = %.4f, mu =" % b0[0], mu0)
print("y - mu:", y - mu0 + 0.0, "  x (y - mu):", x * (y - mu0) + 0.0); g = X.T @ (y - mu0); print("g =", g + 0.0)
print("X'X =", X.T @ X.tolist() if False else (X.T @ X).tolist(), "  H = 3 X'X =", (X.T @ (mu0[:, None] * X)).tolist())
H = X.T @ (mu0[:, None] * X); print("det H = %.1f; H^-1 =" % np.linalg.det(H), np.linalg.inv(H).tolist(), " (1/54)[[15,-9],[-9,9]] =", (np.array([[15, -9], [-9, 9]]) / 54).tolist())
dl = np.linalg.solve(H, g); print("delta =", dl, " = (-2/3, 2/3)")
b1 = b0 + dl; print("new point: b = %.4f, w = %.4f" % tuple(b1)); mu1 = np.exp(X @ b1); print("new means:", mu1)
print("unit deviances before:", dP(y, mu0), " D0 = %.4f" % dP(y, mu0).sum()); print("unit deviances after: ", dP(y, mu1), " D1 = %.4f" % dP(y, mu1).sum())
# the whole iteration, from the start
b = b0.copy()
for k in range(1, 6):
    mu_ = np.exp(X @ b); g_ = X.T @ (y - mu_); H_ = X.T @ (mu_[:, None] * X)
    print("iterate %d: b = %.4f w = %.4f D = %.5f max|g| = %.1e" % (k - 1, b[0], b[1], dP(y, mu_).sum(), abs(g_).max())); b = b + np.linalg.solve(H_, g_)
print("balance at the first step: sum(y - mu) = %.1f, sum x (y - mu) = %.1f ; sum(y) = 9 and sum(mu0) = 9" % ((y - mu0).sum(), (x * (y - mu0)).sum()))
print("least-squares line of the data: slope %.1f intercept %.1f -> divided by 3: %.4f, %.4f" % (2.0, 1.0, 2 / 3, 1 / 3))

print("\n== slips used in the instructor notes")
y0 = np.array([2., 0, 3]); print("14A.1: sum of the two non-zero unit deviances only = %.4f (the zero count forgotten)" % (dP(2., 5 / 3) + dP(3., 5 / 3)))
H1 = X.T @ X; dl1 = np.linalg.solve(H1, g); b1s = b0 + dl1; print("14A.3: with M = I (the factor mu forgotten): H = X'X = %s, delta = %s, point (%.4f, %.4f), D = %.3f (the start has %.3f)" % (H1.tolist(), np.round(dl1, 4), b1s[0], b1s[1], dP(y, np.exp(X @ b1s)).sum(), dP(y, mu0).sum()))
