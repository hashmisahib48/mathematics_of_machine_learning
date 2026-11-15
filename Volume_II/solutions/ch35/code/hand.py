# Hand checks for Problems 35A.1 - 35A.3 (every number quoted in the solutions)
import warnings
import numpy as np
from sklearn.cross_decomposition import PLSRegression
np.set_printoptions(precision=4, suppress=True)

print("== 35A.1  X rows (-3,-1), (-1,-2), (1,2), (3,1); y = (-1, 0, 1, 0)")
X = np.array([[-3., -1], [-1, -2], [1, 2], [3, 1]]); y = np.array([-1., 0, 1, 0])
print("column means:", X.mean(0), y.mean(), " x1.y =", X[:, 0] @ y, " x2.y =", X[:, 1] @ y)
g = X.T @ y; w1 = g / np.linalg.norm(g); print("X^T y =", g, " norm =", np.linalg.norm(g), " w1 =", w1)
t1 = X @ w1; tt = t1 @ t1; print("t1 =", t1, " t1.t1 =", tt)
print("X^T t1 =", X.T @ t1, " y.t1 =", y @ t1)
p1 = X.T @ t1 / tt; q1 = y @ t1 / tt; print("p1 =", p1, " q1 =", q1, " 5/26 =", 5 / 26)
B1 = w1 * q1; print("B after one component =", B1, " (4/26, 3/26 =", 4 / 26 * 1, 3 / 26, "with w q = 0.8*5/26, 0.6*5/26)")
X1 = X - np.outer(t1, p1); y1 = y - q1 * t1
print("deflated X:\n", X1, "\ndeflated y:", y1)
print("check t1 . X1 =", t1 @ X1, " t1 . y1 =", t1 @ y1)
print("sse after one component:", (y1 ** 2).sum(), " (y - X B1)^2 =", ((y - X @ B1) ** 2).sum())
g2 = X1.T @ y1; w2 = g2 / np.linalg.norm(g2); print("X1^T y1 =", g2, " norm =", np.linalg.norm(g2), " w2 =", w2)
t2 = X1 @ w2; tt2 = t2 @ t2; p2 = X1.T @ t2 / tt2; q2 = y1 @ t2 / tt2
print("t2 =", t2, " t2.t2 =", tt2, " p2 =", p2, " q2 =", q2, " t1.t2 =", t1 @ t2)
Wm = np.c_[w1, w2]; Pm = np.c_[p1, p2]; PW = Pm.T @ Wm; R = Wm @ np.linalg.inv(PW)
print("P^T W =\n", PW, "\nR =\n", R)
B2 = R @ np.array([q1, q2]); print("B (two components) =", B2)
ols = np.linalg.solve(X.T @ X, X.T @ y); print("X^T X =", (X.T @ X).tolist(), " det =", np.linalg.det(X.T @ X), " least squares:", ols)
print("sse with two components:", ((y - X @ B2) ** 2).sum(), " y2 residual:", y1 - q2 * t2, " t1.y2, t2.y2 =", t1 @ (y1 - q2 * t2), t2 @ (y1 - q2 * t2))
print("scikit-learn PLSRegression(2, scale=False).coef_ =", PLSRegression(2, scale=False).fit(X, y).coef_.ravel(),
      " (1):", PLSRegression(1, scale=False).fit(X, y).coef_.ravel())
print("third component would need X2^T y2 =", (X1 - np.outer(t2, p2)).T @ (y1 - q2 * t2))

print("\n== 35A.2  chapter data (§35.1): X^T y = (3, 4)")
Xc = np.array([[-3., 1], [-2, -1], [2, 1], [3, -1]]); yc = np.array([0., -1, 2, -1])
print("X^T y =", Xc.T @ yc)
for w in ((1, 0), (0, 1), (0.6, 0.8)):
    w = np.array(w); print("w =", w, " t =", Xc @ w, " t.y =", (Xc @ w) @ yc)
th = np.degrees(np.arctan2(4, 3)); print("angle of X^T y: %.2f deg; t.y at that angle = 5 cos(0) = %.4f" % (th, 5 * np.cos(0)))
for a in (0, 30, 53.13, 90, 120, 143.13):
    wv = np.array([np.cos(np.radians(a)), np.sin(np.radians(a))]); print("  theta %6.2f: t.y = %7.4f  5 cos(theta - 53.13) = %7.4f" % (a, (Xc @ wv) @ yc, 5 * np.cos(np.radians(a - 53.13))))

print("\n== 35A.3  back to the units of the data")
rng = np.random.default_rng(3); n = 40
Xr = np.c_[10 + 2 * rng.standard_normal(n), 5 + 4 * rng.standard_normal(n)]; yr = 20 + 8 * (0.5 * (Xr[:, 0] - 10) / 2 + 0.25 * (Xr[:, 1] - 5) / 4) + rng.standard_normal(n)
m = PLSRegression(2).fit(Xr, yr)
B = (m.x_rotations_ @ m.y_loadings_.T).ravel(); sx = Xr.std(0, ddof=1); sy = yr.std(ddof=1)
print("random check: coef_ =", m.coef_.ravel(), " s_y B_j / s_j =", sy * B / sx, " intercept_ =", m.intercept_, " mean y =", yr.mean())
Bs = np.array([0.5, 0.25]); sx = np.array([2., 4.]); sy = 8.; xm = np.array([10., 5.]); ym = 20.
coef = sy * Bs / sx; print("coef_ =", coef, " intercept_ =", ym)
xn = np.array([12., 9.]); print("prediction:", ym + (xn - xm) @ coef, " via scaled route:", ym + sy * (((xn - xm) / sx) @ Bs))
print("terms:", (xn - xm) * coef)

print("\n== slip of 35A.1: B = q1 w1 + q2 w2 instead of q1 r1 + q2 r2")
print("   ", q1 * w1 + q2 * w2)
