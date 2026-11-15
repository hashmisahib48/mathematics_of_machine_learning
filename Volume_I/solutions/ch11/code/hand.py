# Hand checks for Problems 11A.1 - 11A.4 (every number quoted in the solutions)
import numpy as np
np.set_printoptions(precision=4, suppress=True)
dist = np.array([1., 2, 3, 4, 5]); lights = np.array([2., 1, 4, 3, 5]); y = np.array([3., 4, 6, 7, 10])
n = 5
X = np.c_[dist, lights]; Xc = X - X.mean(0); yc = y - y.mean()
v = (Xc ** 2).mean(0); c = Xc.T @ yc / n; G = Xc.T @ Xc / n
soft = lambda z, a: float(np.sign(z) * max(abs(z) - a, 0.0))

print("== common numbers")
print("centred distance:", Xc[:, 0], " centred lights:", Xc[:, 1], " centred y:", yc)
print("v =", v, " c =", c, " mean product x1x2 =", G[0, 1], " means:", X.mean(0), y.mean())
print("least squares, distance only: w =", c[0] / v[0], " b =", y.mean() - c[0] / v[0] * 3, " prediction at 6:", y.mean() + c[0] / v[0] * 3)

print("\n== 11A.1  distance only, alpha = 2.5")
a = 2.5
print("alpha_max =", abs(c[0]))
w = soft(c[0], a) / v[0]; b = y.mean() - w * 3
print("numerator c - alpha =", c[0] - a, " w =", w, " b = 6 - 3w =", b, " prediction at 6 =", b + 6 * w)
print("fitted values:", b + w * dist, " residual sum of squares:", ((y - b - w * dist) ** 2).sum())
print("KKT: |cov(x, residual)| =", abs(Xc[:, 0] @ (yc - w * Xc[:, 0]) / n), " (must equal alpha)")
Sxy, Sxx = Xc[:, 0] @ yc, Xc[:, 0] @ Xc[:, 0]
print("slips: unnormalised sums Sxy =", Sxy, " Sxx =", Sxx, " -> (Sxy - alpha)/Sxx =", (Sxy - a) / Sxx, "; forgetting to divide by v:", c[0] - a)
print("shrinkage per unit: alpha/v =", a / v[0], "  weights at alpha=0,1,2.5,3.4:", [soft(c[0], q) / v[0] for q in (0, 1, 2.5, 3.4)])

print("\n== 11A.2  both features, alpha = 2, two sweeps")
a = 2.0; w = np.zeros(2); r = yc.copy()
for sweep in (1, 2):
    for j in range(2):
        xr = Xc[:, j] @ r / n                       # covariance with the current residual
        z = xr + v[j] * w[j]                        # Eq. 11.3-10
        wn = soft(z, a) / v[j]
        print(f"sweep {sweep} j={j+1}: x'r/n = {xr:.4f}  v w_old = {v[j]*w[j]:.4f}  z = {z:.4f}  soft = {soft(z, a):.4f}  w_new = {wn:.4f}")
        r = r + Xc[:, j] * (w[j] - wn); w[j] = wn
    print("   w =", w, " residual:", r)
print("slip (old residual used for the second feature, z2 = c2 = 3.0): w2 =", soft(c[1], a) / v[1])
print("intercept b = 6 - 3 w1 - 3 w2 =", y.mean() - X.mean(0) @ w)
print("correlation of the lights with the residual:", Xc[:, 1] @ r / n, " of the distance:", Xc[:, 0] @ r / n)
print("lights enter at alpha = 1.4: 0.28 + 0.8*1.4 =", 0.28 + 0.8 * 1.4, " c2 - 1.6 w1 at alpha=2, w1=0.7:", c[1] - 1.6 * 0.7)

print("\n== 11A.3  elastic net, one feature, alpha = 1, rho = 0.5")
a, rho = 1.0, 0.5
l1, l2 = a * rho, a * (1 - rho)
w = soft(c[0], l1) / (v[0] + l2); b = y.mean() - 3 * w
print("lambda1 =", l1, " lambda2 =", l2, " numerator =", c[0] - l1, " denominator =", v[0] + l2, " w =", w, " b =", b)
print("lasso alpha=1: w =", soft(c[0], 1) / v[0], " b =", y.mean() - 3 * soft(c[0], 1) / v[0])
print("L1 part alone (lambda1=0.5, no L2):", soft(c[0], 0.5) / v[0], "  L2 part alone (c/(v+0.5)):", c[0] / (v[0] + 0.5))
print("slip: threshold alpha instead of alpha*rho:", soft(c[0], 1.0) / (v[0] + l2))
print("predictions at 6: enet", b + 6 * w, " lasso", 2.4 + 6 * 1.2)
print("KKT enet: c - v w - lambda2 w =", c[0] - v[0] * w - l2 * w, " (must equal lambda1 =", l1, ")")

print("\n== 11A.4  group soft threshold")
z = np.array([6., 8.]); vv = 2.0; a = 5.0
nz = np.linalg.norm(z); fac = max(1 - a / nz, 0)
print("||z|| =", nz, " factor =", fac, " w =", fac * z / vv)
print("separate lassos:", [soft(q, a) / vv for q in z], " ratio of the group weights:", (fac * z / vv)[0] / (fac * z / vv)[1], " ratio of separate:", soft(z[0], a) / soft(z[1], a))
for al in (2, 5, 8, 10, 12):
    f = max(1 - al / nz, 0); print(f"alpha = {al}: group w = {f * z / vv},  separate = {[soft(q, al) / vv for q in z]}")
print("row zero for alpha >=", nz, "; first target alone is zero from alpha >= 6, second from alpha >= 8")
