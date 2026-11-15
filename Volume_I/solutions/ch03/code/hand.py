# Hand checks for Problems 3A.1 - 3A.4 (every number quoted in the solutions)
import numpy as np
np.set_printoptions(precision=4, suppress=True)
sg = lambda z: 1 / (1 + np.exp(-z))
nd = lambda f, t, h=1e-6: (f(t + h) - f(t - h)) / (2 * h)

print("== 3A.1  f = (3x^2+1)^4, g = x^2 exp(-x), x = 1")
u = 3 * 1 ** 2 + 1; du = 6 * 1; print("u =", u, " du/dx =", du, " 4u^3 =", 4 * u ** 3, " f'(1) =", 4 * u ** 3 * du)
f = lambda t: (3 * t ** 2 + 1) ** 4; g = lambda t: t ** 2 * np.exp(-t)
print("central difference f'(1) =", round(nd(f, 1.0), 2))
print("exp(-1) =", np.exp(-1), " 2x-x^2 =", 2 - 1, " g'(1) =", (2 - 1) * np.exp(-1), " numeric:", round(nd(g, 1.0), 4))

print("\n== 3A.2  sigmoid")
print("sigma(0) =", sg(0.0), " slope(0) =", sg(0.0) * (1 - sg(0.0)))
print("exp(-2) =", np.exp(-2), " 1+exp(-2) =", 1 + np.exp(-2), " sigma(2) =", sg(2.0), " 1-sigma(2) =", 1 - sg(2.0))
print("slope(2) =", sg(2.0) * (1 - sg(2.0)), " numeric:", round(nd(sg, 2.0), 4))
print("0.25/slope(2) =", 0.25 / (sg(2.0) * (1 - sg(2.0))), " slope(2)/0.25 =", sg(2.0) * (1 - sg(2.0)) / 0.25)

print("\n== 3A.3  f = u^2 + uv + 2v^2")
grad = lambda u, v: np.array([2 * u + v, u + 4 * v]); H = np.array([[2., 1], [1, 4]])
print("grad at (1,-1):", grad(1, -1), " f(1,-1) =", 1 - 1 + 2)
print("trace =", np.trace(H), " det =", np.linalg.det(H), " 3 -/+ sqrt(2):", 3 - np.sqrt(2), 3 + np.sqrt(2))
print("eig:", np.linalg.eigvalsh(H), " stationary point:", np.linalg.solve(H, np.zeros(2)), " f there = 0")
print("f(1,-1) via H: 0.5 x'Hx =", 0.5 * np.array([1, -1]) @ H @ np.array([1, -1]))

print("\n== 3A.4  one descent step, delivery data")
x = np.array([1., 2, 3, 4, 5]); y = np.array([3., 4, 6, 7, 10]); X = np.c_[np.ones(5), x]
J = lambda v: float(np.sum((y - X @ v) ** 2)); v = np.array([1., 2.])
r = y - X @ v; print("residuals:", r, " squares:", r ** 2, " J =", J(v))
print("sum r =", r.sum(), " sum x r =", x @ r, " x*r:", x * r)
gr = -2 * X.T @ r; print("gradient:", gr, " norm^2 =", gr @ gr)
eta = 0.01; vn = v - eta * gr; print("new point:", vn)
rn = y - X @ vn; print("new residuals:", rn, " squares:", rn ** 2, " J =", J(vn))
Hs = 2 * X.T @ X; print("Hessian:", Hs.tolist(), " H g =", Hs @ gr, " g'Hg =", gr @ Hs @ gr)
print("first-order change -eta|g|^2 =", -eta * gr @ gr, " curvature term 0.5 eta^2 g'Hg =", 0.5 * eta ** 2 * gr @ Hs @ gr)
print("sum =", -eta * gr @ gr + 0.5 * eta ** 2 * gr @ Hs @ gr, " actual change =", J(vn) - J(v))
print("J(v) - J(vn) =", J(v) - J(vn))
for e in (0.001, 0.005, 0.01, 0.015, 0.02, 0.0169):
    print(f"  eta = {e}: J after = {J(v - e * gr):.3f}")
print("largest eta that lowers J along -g: 2|g|^2/(g'Hg) =", 2 * gr @ gr / (gr @ Hs @ gr), "  2/lambda_max =", 2 / np.linalg.eigvalsh(Hs)[1])
print("\nslips and ratios quoted in the notes: 4u^3 =", 4 * 4 ** 3, " 1536/256 =", 1536 / 256, " (2x+x^2)e^-1 =", 3 * np.exp(-1),
      " slope(6) =", sg(6.0) * (1 - sg(6.0)), " 0.25/slope(6) =", 0.25 / (sg(6.0) * (1 - sg(6.0))))
