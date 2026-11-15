# Numbers quoted in the model answers to the questions of Chapter 30.
import sys, os, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../ch30/code')
import numpy as np, warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)
from scipy.stats import multivariate_normal
from sklearn.gaussian_process import GaussianProcessRegressor as GPR, GaussianProcessClassifier as GPC
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel, DotProduct
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import LeaveOneOut
sig = lambda z: 1 / (1 + np.exp(-z))
TX = np.array([1., 2, 3, 5, 6, 8])[:, None]; TY = np.array([0.2, 1.1, 1.5, 0.4, -0.6, 0.3])

print("== Q1  prior mean zero, far from the data (readings + 100, x* = 30, alpha = 0.01)")
Y = TY + 100
for nm, ker, nrm in [("C*RBF, normalize_y=False", C(1.0) * RBF(1.5), False), ("C*RBF, normalize_y=True", C(1.0) * RBF(1.5), True),
                     ("C(100^2 fixed)+C*RBF, no normalize", C(1e4, "fixed") + C(1.0) * RBF(1.5), False)]:
    g = GPR(ker, alpha=0.01, normalize_y=nrm, random_state=0, n_restarts_optimizer=3).fit(TX, Y)
    m, s = g.predict([[30.0]], return_std=True); print(f"{nm:38s} mean {m[0]:8.3f}  std {s[0]:.3f}   kernel_ {g.kernel_}")
xs = np.arange(1, 9.0)[:, None]; yl = 2 + 0.5 * xs[:, 0] + 0.1 * np.random.default_rng(0).normal(size=8)
for nm, ker in [("C*RBF", C(1.0) * RBF(2.0)), ("C*RBF + DotProduct", C(1.0) * RBF(2.0) + DotProduct(1.0))]:
    g = GPR(ker, alpha=0.01, normalize_y=True, random_state=0, n_restarts_optimizer=3).fit(xs, yl)
    print(f"linear trend, kernel {nm:20s} prediction at x=20: {g.predict([[20.0]])[0]:.3f}  (the line gives {2 + 0.5 * 20:.1f}); at x=30: {g.predict([[30.0]])[0]:.3f} (line {2 + 15:.1f})")

print("\n== Q2  what the process adds to kernel ridge regression")
ell, s2 = 1.3672, 0.01
g = GPR(RBF(ell, "fixed"), alpha=s2, optimizer=None).fit(TX, TY)           # amplitude 1
kr = KernelRidge(kernel="rbf", gamma=1 / (2 * ell ** 2), alpha=s2).fit(TX, TY)
grid = np.linspace(0, 12, 121)[:, None]
print("max |GP mean - KernelRidge| on [0,12]:", float(np.abs(g.predict(grid) - kr.predict(grid)).max()))
g2 = GPR(C(1.0) * RBF(1.0), alpha=0.01, n_restarts_optimizer=5, random_state=0).fit(TX, TY)
for xq in (4.0, 9.0, 12.0):
    m, s = g2.predict([[xq]], return_std=True); print(f"fitted GP at x={xq:4.1f}: mean {m[0]:.3f}  std {s[0]:.3f}")
print("kernel_", g2.kernel_, " LML", round(g2.log_marginal_likelihood_value_, 3))
rng = np.random.default_rng(0); n = 400; X = rng.uniform(0, 10, (n, 2)); y = np.sin(X[:, 0]) + 0.1 * rng.normal(size=n)
cnt = [0]
def opt(obj, th0, bounds):
    from scipy.optimize import minimize
    def f(t): cnt[0] += 1; return obj(t, eval_gradient=True)
    r = minimize(f, th0, jac=True, bounds=bounds, method="L-BFGS-B"); return r.x, r.fun
ker = C(1.0) * RBF([1.0, 1.0]) + WhiteKernel(0.1)
t0 = time.perf_counter(); GPR(ker, optimizer=opt, random_state=0).fit(X, y); t1 = time.perf_counter() - t0
t0 = time.perf_counter(); GPR(ker, optimizer=None).fit(X, y); t2 = time.perf_counter() - t0
print(f"n={n}: optimiser called the likelihood {cnt[0]} times, fit {t1:.2f} s against {t2:.3f} s for one factorisation (ratio {t1 / t2:.0f})")

print("\n== Q3  marginal likelihood against leave-one-out cross-validation (14 readings of Section 30.11)")
rng = np.random.default_rng(1); dx = np.sort(rng.uniform(0, 10, 14)); dy = np.sin(0.9 * dx) + 0.35 * rng.normal(size=14); DX = dx[:, None]
tg = np.linspace(0, 10, 400)[:, None]
for nm, (l0, n0) in [("peak 1", (0.3, 0.05 ** 2)), ("peak 2", (1.4, 0.3 ** 2))]:
    gk = GPR(C(1.0, "fixed") * RBF(l0) + WhiteKernel(n0), alpha=1e-10, random_state=0).fit(DX, dy)
    pr = gk.kernel_.get_params(); l = [v for k, v in pr.items() if k.endswith("length_scale")][0]; nz = [v for k, v in pr.items() if k.endswith("noise_level")][0]
    loo = []; lpd = []
    for i in range(14):
        m = np.arange(14) != i
        gi = GPR(C(1.0, "fixed") * RBF(l, "fixed"), alpha=nz, optimizer=None).fit(DX[m], dy[m]); mu, sd = gi.predict(DX[i:i + 1], return_std=True); sd = np.sqrt(sd ** 2 + nz)
        loo.append(mu[0] - dy[i]); lpd.append(-0.5 * np.log(2 * np.pi * sd[0] ** 2) - 0.5 * (loo[-1] / sd[0]) ** 2)
    ge = GPR(C(1.0, "fixed") * RBF(l, "fixed"), alpha=nz, optimizer=None).fit(DX, dy)
    err = np.sqrt(np.mean((ge.predict(tg) - np.sin(0.9 * tg[:, 0])) ** 2))
    print(f"{nm}: ell {l:.3f}  sigma_n {np.sqrt(nz):.3f}  LML {gk.log_marginal_likelihood_value_:.3f}  LOO rmse {np.sqrt(np.mean(np.square(loo))):.3f}  LOO log density {sum(lpd):.2f}  rmse to the true sin on [0,10] {err:.3f}")
S2 = 0.1
x2 = np.array([0.0, 2.0]); y2 = np.array([1.2, 0.4])
print("two readings of Example 30.6-1: the likelihood and the leave-one-out errors at ell = 0.3, 1, 3")
for l in (0.3, 1.0, 3.0):
    Sm = np.exp(-(x2[:, None] - x2[None, :]) ** 2 / (2 * l ** 2)) + S2 * np.eye(2)
    print(f"  ell {l}: LML {-0.5 * y2 @ np.linalg.solve(Sm, y2) - 0.5 * np.log(np.linalg.det(Sm)) - np.log(2 * np.pi):.4f}")
    e = []
    for i in range(2):
        j = 1 - i; kij = np.exp(-(x2[i] - x2[j]) ** 2 / (2 * l ** 2)); e.append(kij / (1 + S2) * y2[j] - y2[i])
    print(f"  ell {l}: LOO errors {np.round(e, 3)}  rmse {np.sqrt(np.mean(np.square(e))):.3f}")

print("\n== Q5  Laplace mode against the exact posterior (Example 30.10-1: x = -1, 1; labels 0, 1; x* = 2)")
rho = np.exp(-2.0); K = np.array([[1, rho], [rho, 1.0]]); g1 = np.linspace(-6, 6, 801); F1, F2 = np.meshgrid(g1, g1, indexing="ij")
logp = multivariate_normal(np.zeros(2), K).logpdf(np.dstack([F1, F2])) + np.log(1 - sig(F1)) + np.log(sig(F2)); w = np.exp(logp - logp.max()); w /= w.sum()
m1, m2 = (w * F1).sum(), (w * F2).sum(); v1 = (w * F1 ** 2).sum() - m1 ** 2; v2 = (w * F2 ** 2).sum() - m2 ** 2
print(f"exact posterior mean of (f1, f2): ({m1:.5f}, {m2:.5f}); Laplace mode (-0.35615, 0.35615); exact sd ({np.sqrt(v1):.4f}, {np.sqrt(v2):.4f})")
W = np.diag(sig(np.array([-0.35615, 0.35615])) * (1 - sig(np.array([-0.35615, 0.35615]))))
Sig = np.linalg.inv(np.linalg.inv(K) + W); print("Laplace covariance diag:", np.sqrt(np.diag(Sig)).round(4), " correlation", round(Sig[0, 1] / np.sqrt(Sig[0, 0] * Sig[1, 1]), 4))
ks = np.array([np.exp(-4.5), np.exp(-0.5)]); a = np.linalg.solve(K, ks); vs = 1 - ks @ a
gh, gw = np.polynomial.hermite_e.hermegauss(60); gw = gw / gw.sum()
def prob(mean_f_given_f, var_star): return sum(gw[j] * sig(mean_f_given_f + np.sqrt(var_star) * gh[j]) for j in range(60))
mu_f = a[0] * F1 + a[1] * F2
exact = (w * prob(mu_f, vs)).sum()
pl = GPC(C(1.0, "fixed") * RBF(1.0, "fixed"), optimizer=None).fit(np.array([[-1.0], [1.0]]), [0, 1]).predict_proba([[2.0]])[0, 1]
print(f"P(y*=1) at x*=2: exact posterior {exact:.4f}; Laplace approximation (GaussianProcessClassifier) {pl:.4f}")
