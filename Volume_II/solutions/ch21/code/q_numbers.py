# Numbers quoted in the model answers to the questions for discussion (Q1 - Q4)
import warnings, numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics.pairwise import rbf_kernel, sigmoid_kernel
np.set_printoptions(precision=4, suppress=True)

print("== Q1  a finite set of numbers instead of the weight vector")
X3 = np.array([[-1.], [0.], [1.]]); y3 = np.array([1., 0., 1.])
K3 = rbf_kernel(X3, gamma=0.5); a3 = np.linalg.solve(K3 + 0.1 * np.eye(3), y3)
print("three readings, RBF: dual coefficients", a3, " (3 numbers for a weight vector of infinitely many)")
rng = np.random.default_rng(1); X = rng.normal(size=(30, 2)); y = X[:, 0] * X[:, 1] + X[:, 0] ** 2 + 0.1 * rng.normal(size=30)
Xn = rng.normal(size=(4, 2))
def phi(M):
    a, b, s = M[:, 0], M[:, 1], np.sqrt(2); return np.c_[np.ones(len(M)), s * a, s * b, a * a, s * a * b, b * b]
kr = KernelRidge(alpha=0.7, kernel="poly", degree=2, gamma=1.0, coef0=1.0).fit(X, y)
rd = Ridge(alpha=0.7, fit_intercept=False).fit(phi(X), y)
print("degree-2 kernel, 30 samples: dual_coef_ has", kr.dual_coef_.shape[0], "numbers, the explicit weight vector has", rd.coef_.shape[0])
print("  w = Phi^T a equals the weights of Ridge on the six columns:", np.allclose(phi(X).T @ kr.dual_coef_, rd.coef_),
      "; largest difference of predictions %.1e" % abs(kr.predict(Xn) - rd.predict(phi(Xn))).max())
rb = KernelRidge(alpha=0.7, kernel="rbf", gamma=0.5).fit(X, y)
print("RBF kernel, same 30 samples: dual_coef_ has", rb.dual_coef_.shape[0], "numbers; solving needs a", len(X), "x", len(X), "system")

print("\n== Q2  kernel ridge regression and nearest neighbours on the three readings")
H = rbf_kernel(np.array([[0.5], [4.0]]), X3, gamma=0.5) @ np.linalg.inv(K3 + 0.1 * np.eye(3))
for q, row in zip((0.5, 4.0), H):
    print("query %.1f: weights on the three readings y = (1, 0, 1):" % q, row, " sum %.4f  prediction %.4f" % (row.sum(), row @ y3))
for k in (1, 2):
    m = KNeighborsRegressor(k).fit(X3, y3); print("k-NN, k = %d: f(0.5) = %.3f   f(4) = %.3f" % (k, m.predict([[0.5]])[0], m.predict([[4.0]])[0]))
r4 = np.random.default_rng(4); xs = np.sort(r4.uniform(0, 6, 28)); ys = np.sin(xs) + 0.25 * r4.normal(size=28)
Ks = rbf_kernel(xs[:, None], gamma=0.5); Hs = rbf_kernel(np.array([[3.0]]), xs[:, None], gamma=0.5) @ np.linalg.inv(Ks + 0.1 * np.eye(28))
print("28 sine samples, query x = 3: %d weights, %d negative, sum %.4f, largest %.3f, smallest %.3f, sum of |weights| %.3f"
      % (Hs.size, (Hs < 0).sum(), Hs.sum(), Hs.max(), Hs.min(), np.abs(Hs).sum()))
print("k-NN weights (k = 5, uniform): five weights 0.2, none negative, sum 1")

print("\n== Q3  the sigmoid kernel")
X8 = np.random.default_rng(0).normal(size=(8, 3)); K8 = sigmoid_kernel(X8, gamma=1.0, coef0=0.0)
ev = np.linalg.eigvalsh(K8); print("eigenvalues of the 8 x 8 Gram matrix:", ev, " negative:", int((ev < 0).sum()))
y8 = np.sin(X8[:, 0]) + X8[:, 1] ** 2
for al in (0.1, 0.5, 1.0, 1.5):
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always"); m = KernelRidge(alpha=al, kernel="sigmoid", gamma=1.0, coef0=0.0).fit(X8, y8)
    lam = np.linalg.eigvalsh(K8 + al * np.eye(8)).min()
    print("alpha = %.1f: smallest eigenvalue of K + alpha I = %.3f; warnings: %s" % (al, lam, [str(w.message)[:40] for w in wl]))
al = 1.5; a_sol = np.linalg.solve(K8 + al * np.eye(8), y8); J = lambda a: ((y8 - K8 @ a) ** 2).sum() + al * a @ K8 @ a
w, V = np.linalg.eigh(K8); v = V[:, 0]
print("alpha = 1.5 (no warning): J(a) of Eq. 21.4-5 at the solution of (K + alpha I)a = y: %.4f" % J(a_sol))
print("  curvature of J along the eigenvector of lambda = %.4f is proportional to lambda (lambda + alpha) = %.4f (negative)" % (w[0], w[0] * (w[0] + al)))
print("  J at a + t v for t = 1, 10, 100: %.1f, %.1f, %.1f   (falls without limit: the solution is not a minimum)" % tuple(J(a_sol + t * v) for t in (1, 10, 100)))
print("  K lambda(lambda + alpha) for all eigenvalues:", w * (w + al))

print("\n== Q4  two panels of Figure 21.9-1 with similar df")
grid = np.linspace(0, 6, 300)
def fit(g, a):
    m = KernelRidge(alpha=a, kernel="rbf", gamma=g).fit(xs[:, None], ys - ys.mean()); p = m.predict(grid[:, None]) + ys.mean()
    ev = np.linalg.eigvalsh(rbf_kernel(xs[:, None], gamma=g)); df = (ev / (ev + a)).sum()
    train = np.sqrt(np.mean((m.predict(xs[:, None]) + ys.mean() - ys) ** 2))
    return p, df, np.sqrt(np.mean((p - np.sin(grid)) ** 2)), np.abs(m.dual_coef_).max(), np.linalg.cond(rbf_kernel(xs[:, None], gamma=g) + a * np.eye(28)), train, np.abs(p).max()
for g, a in ((5, 0.1), (0.5, 1e-5), (0.5, 0.1)):
    p, df, rm, am, cn, tr, mx = fit(g, a)
    print("gamma = %-4g alpha = %-6g df = %5.2f  rmse from sin = %.3f  training rmse = %.3f  max|f| = %.2f  largest |a_i| = %9.1f  condition number of K + alpha I = %.1e"
          % (g, a, df, rm, tr, mx, am, cn))
print("noise standard deviation of the data: 0.25")
# distance of the fit from the truth in the gaps between data points versus at the data points
for g, a in ((5, 0.1), (0.5, 1e-5)):
    m = KernelRidge(alpha=a, kernel="rbf", gamma=g).fit(xs[:, None], ys - ys.mean()); mid = (xs[1:] + xs[:-1]) / 2
    print("gamma = %g alpha = %g: rmse from sin at the data %.3f, midway between neighbouring data %.3f" % (g, a, np.sqrt(np.mean((m.predict(xs[:, None]) + ys.mean() - np.sin(xs)) ** 2)), np.sqrt(np.mean((m.predict(mid[:, None]) + ys.mean() - np.sin(mid)) ** 2))))
