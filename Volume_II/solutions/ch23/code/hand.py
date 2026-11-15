# Hand checks for the Chapter 23 solutions (23A.1 - 23A.4); every number quoted in the text comes from here.
import numpy as np
from sklearn.svm import SVR, NuSVR
print("== 23A.1")
r = np.array([-1.5, -0.4, 0.2, 0.9, 2.1]); eps = 0.5; C = 2.0
L = np.maximum(0, np.abs(r) - eps)
print("L =", L.round(2), " inside:", int((np.abs(r) <= eps).sum()), " sum L =", L.sum().round(2),
      " sum|r| =", np.abs(r).sum().round(2), " sum r^2 =", (r**2).sum().round(2))
print("0.5 ||w||^2 =", 0.5 * 4, " C sum L =", C * L.sum().round(2), " J =", 0.5 * 4 + C * L.sum())
print("square loss of r = 4: 16; L_0.5(4) =", 4 - 0.5)
print("== 23A.2")
x = np.array([1., 3.]); y = np.array([1., 4.]); eps = 0.5; C = 1.0
A = np.array([[1, 1.], [3, 1.]]); w, b = np.linalg.solve(A, [y[0] + eps, y[1] - eps])   # f(1)=y1+eps, f(3)=y2-eps
beta2 = w / (x[1] - x[0]); beta = np.array([-beta2, beta2])
print("w, b =", w, b, " beta =", beta, " f(2) =", w * 2 + b, " residuals", (y - (w * x + b)))
print("sum beta =", beta.sum(), " w = sum beta x =", beta @ x, " |beta| <= C:", bool(np.abs(beta).max() <= C))
primal = 0.5 * w**2 + C * np.maximum(0, np.abs(y - (w * x + b)) - eps).sum()
dual = -0.5 * (beta @ x)**2 + y @ beta - eps * np.abs(beta).sum()
print("primal =", primal, " dual =", dual, " pieces of the dual:", -0.5 * (beta @ x)**2, y @ beta, -eps * np.abs(beta).sum())
m = SVR(kernel="linear", C=C, epsilon=eps, tol=1e-12).fit(x[:, None], y)
print("SVR:", m.coef_[0, 0], m.intercept_[0], m.dual_coef_[0], m.support_)
print("== 23A.3")
r = np.array([0.1, -0.4, 0.9, -1.2, 0.4, 0.0]); eps = 0.4; C = 2.0
xi = np.maximum(0, r - eps); xs = np.maximum(0, -r - eps)
print("xi =", xi.round(2), " xi* =", xs.round(2), " sum =", (xi + xs).sum().round(2))
print("0.5 ||w||^2 =", 0.5 * 2.25, " C sum =", C * (xi + xs).sum().round(2), " J =", 0.5 * 2.25 + C * (xi + xs).sum().round(2))
print("kind:", ["inside" if abs(v) < eps - 1e-12 else "edge" if abs(abs(v) - eps) < 1e-12 else "outside" for v in r])
print("== 23A.4")
rng = np.random.default_rng(0); n, nu, C = 50, 0.3, 2.0
X = rng.uniform(0, 6, (n, 1)); yv = np.sin(X[:, 0]) + 0.3 * rng.normal(size=n)
m = NuSVR(nu=nu, C=C, gamma=0.5, tol=1e-8).fit(X, yv)
res = yv - m.predict(X)          # eps is not stored: recover it from the free SVs
beta = np.zeros(n); beta[m.support_] = m.dual_coef_[0]
free = (np.abs(beta) > 1e-8) & (np.abs(beta) < C - 1e-8); bound = np.abs(beta) >= C - 1e-8
eps_fit = np.abs(res[free]).mean()
out = np.abs(res) > eps_fit + 1e-6
print("nu n =", nu * n, " outside the tube:", int(out.sum()), " support vectors:", len(m.support_), " bound:", int(bound.sum()), " free:", int(free.sum()))
print("sum |beta| =", np.abs(beta).sum().round(6), " C nu n =", C * nu * n, " eps found =", eps_fit.round(4))
print("12 outside: 12 C =", 12 * C, " rest =", C * nu * n - 12 * C, " each free < C = 2 -> free >", (C * nu * n - 12 * C) / C, "-> at least", int((C * nu * n - 12 * C) // C + 1), " SV >=", 12 + int((C * nu * n - 12 * C) // C + 1))
