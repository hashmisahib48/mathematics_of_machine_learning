# Hand checks for Problems 16A.1 - 16A.4 (every number quoted in the solutions)
import numpy as np
np.set_printoptions(precision=4, suppress=True)
sig = lambda z: 1 / (1 + np.exp(-z))

print("== 16A.1  z = -2 + 0.5 x")
for x in (2, 4, 8):
    z = -2 + 0.5 * x
    print(f"x = {x}: z = {z:5.2f}  e^-z = {np.exp(-z):.4f}  p = 1/(1+e^-z) = {sig(z):.4f}  odds e^z = {np.exp(z):.4f}")
print("ln 9 =", np.log(9), " x = (ln 9 + 2)/0.5 =", (np.log(9) + 2) / 0.5, " check p =", sig(-2 + 0.5 * ((np.log(9) + 2) / 0.5)))
print("odds ratio per unit of x: e^0.5 =", np.exp(0.5))

print("\n== 16A.2  labels 1,0,1,0 with p = 0.9, 0.2, 0.6, 0.5")
y = np.array([1, 0, 1, 0]); p = np.array([0.9, 0.2, 0.6, 0.5])
prob_true = np.where(y == 1, p, 1 - p)
print("probability given to the true label:", prob_true)
li = -np.log(prob_true); print("loss each:", li, " J =", li.sum())
print("same by Eq. 16.3-5 first line:", -(y * np.log(p) + (1 - y) * np.log(1 - p)))
print("same by the margin form ln(1 + e^{-s z}):", np.log1p(np.exp(-(2 * y - 1) * np.log(p / (1 - p)))))
print("model that says 0.5: 4 ln 2 =", 4 * np.log(2), " difference", 4 * np.log(2) - li.sum())
print("ln values: ln0.9, ln0.8, ln0.6, ln0.5 =", np.log([0.9, 0.8, 0.6, 0.5]))

print("\n== 16A.3  study data at b = w = 0")
x = np.arange(1., 9); yy = np.array([0, 0, 0, 1, 0, 1, 1, 1.]); X = np.c_[np.ones(8), x]
J = lambda v: float(np.sum(np.logaddexp(0, X @ v) - yy * (X @ v)))
g = X.T @ (0.5 - yy); H = 0.25 * X.T @ X
print("g =", g, " H =", H.tolist(), " J(0) =", J(np.zeros(2)), " 8 ln 2 =", 8 * np.log(2))
for eta in (0.02, 0.05):
    v = -eta * g
    print(f"eta = {eta}: (b, w) = {v + 0.0}  J = {J(v):.4f}")
lam, V = np.linalg.eigh(H); print("eigenvalues of H:", lam, " 2/lam_max =", 2 / lam[-1], " eigenvector of lam_max:", V[:, -1])
print("trace", np.trace(H), " det", np.linalg.det(H), " lam from (tr +- sqrt(tr^2 - 4 det))/2:",
      (np.trace(H) + np.sqrt(np.trace(H) ** 2 - 4 * np.linalg.det(H))) / 2)
print("g.g =", g @ g, " g'Hg =", g @ H @ g, " best step along -g in the quadratic model: g.g/g'Hg =", (g @ g) / (g @ H @ g))
print("component of g along the top eigenvector:", V[:, -1] @ g, " factor |1 - eta lam_max| for eta 0.02, 0.05:",
      abs(1 - 0.02 * lam[-1]), abs(1 - 0.05 * lam[-1]))
for eta in (0.02, 0.05):
    print(f"quadratic model of J at eta={eta}: {J(np.zeros(2)) - eta * (g @ g) + 0.5 * eta**2 * (g @ H @ g):.4f}")
d = -np.linalg.solve(H, g); print("Newton step d =", d, " J(d) =", J(d), " H^-1 =", np.linalg.inv(H).tolist(), " det =", np.linalg.det(H))
print("steps of gradient descent with eta = 0.02 needed to reach J < 2.9049 (the Newton value):", end=" ")
v = np.zeros(2); k = 0
while J(v) >= 2.9049 and k < 100000: v = v - 0.02 * (X.T @ (sig(X @ v) - yy)); k += 1
print(k, " at (b, w) =", v)

print("\n== 16A.4  scores (0, 1, 3), true class 2")
f = np.array([0., 1, 3]); e = np.exp(f); P = e / e.sum()
print("exp:", e, " sum:", e.sum(), " P:", P, " sum P:", P.sum())
Y = np.array([0, 1, 0.]); print("loss -ln P2 =", -np.log(P[1]), " P - Y =", P - Y, " sum:", (P - Y).sum())
f0 = np.array([2., 1, 0]); P0 = np.exp(f0) / np.exp(f0).sum(); print("Example 16.4-1: P =", P0, " loss =", -np.log(P0[1]))
print("loss through logsumexp - f_y:", np.log(e.sum()) - f[1], " ratio of losses:", -np.log(P[1]) / -np.log(P0[1]))
print("score of the true class minus the largest score: here", f[1] - f.max(), " in Example 16.4-1", f0[1] - f0.max())

print("\n== 16A.3  table of the loss of each student after the two gradient steps")
for eta in (0.02, 0.05):
    v = -eta * g; z = X @ v; pp = sig(z); li = np.logaddexp(0, z) - yy * z
    print(f"eta = {eta}: w = {v[1]:.2f}")
    for xi, yi_, zi, pi, l in zip(x, yy, z, pp, li): print(f"   x={xi:.0f} y={yi_:.0f}  z={zi:5.2f}  p={pi:.4f}  loss={l:.4f}")
    print("   sum of losses:", li.sum())
print("p at b = w = 0: 0.5 for all; loss each ln 2 =", np.log(2))
print("sqrt(53^2 - 4*21) =", np.sqrt(53**2 - 84), " (53 + that)/2 =", (53 + np.sqrt(53**2 - 84)) / 2, " (53 - that)/2 =", (53 - np.sqrt(53**2 - 84)) / 2)

print("\n== 16A.2  extra checks")
print("product of the probabilities of the true labels:", prob_true.prod(), " -ln(product) =", -np.log(prob_true.prod()), " uniform model: 0.5^4 =", 0.5**4)
print("wrong: -ln p for every sample regardless of the label:", (-np.log(p)).round(4), " sum", (-np.log(p)).sum().round(4))
print("odds ratio check, Example 16.6-1: e^1.2823 =", np.exp(1.2823))
