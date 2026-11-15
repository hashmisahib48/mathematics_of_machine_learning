# Checks of the statements of Chapter 30 that are not shown in a listing.
import time, warnings, json
import numpy as np
from scipy.special import kv, gamma as G
from scipy.integrate import quad
from scipy.special import expit
from scipy.stats import norm
from sklearn.gaussian_process import (GaussianProcessRegressor as GPR,
    GaussianProcessClassifier as GPC)
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
    ExpSineSquared, DotProduct, WhiteKernel, ConstantKernel as C)
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
OUT = {}
a = np.array([[0.3], [1.1]]); b = np.array([[2.0]]); d = abs(a[:, 0] - 2.0)
def chk(name, k, f):
    ok = np.allclose(k(a, b)[:, 0], f); print(f"{name:28s} formula ok: {ok}"); assert ok
l = 1.7
chk("RBF", RBF(l), np.exp(-d**2 / (2*l**2)))
chk("Matern 0.5", Matern(l, nu=0.5), np.exp(-d/l))
chk("Matern 1.5", Matern(l, nu=1.5), (1+np.sqrt(3)*d/l)*np.exp(-np.sqrt(3)*d/l))
chk("Matern 2.5", Matern(l, nu=2.5),
    (1+np.sqrt(5)*d/l+5*d**2/(3*l**2))*np.exp(-np.sqrt(5)*d/l))
nu = 0.8; z = np.sqrt(2*nu)*d/l
chk("Matern 0.8 (Bessel)", Matern(l, nu=nu), 2**(1-nu)/G(nu)*z**nu*kv(nu, z))
chk("Matern inf = RBF", Matern(l, nu=np.inf), np.exp(-d**2/(2*l**2)))
al = 0.9; chk("RationalQuadratic", RationalQuadratic(l, al), (1+d**2/(2*al*l**2))**(-al))
p = 2.5; chk("ExpSineSquared", ExpSineSquared(l, p), np.exp(-2*np.sin(np.pi*d/p)**2/l**2))
chk("DotProduct", DotProduct(0.7), 0.7**2 + (a*b).sum(1))
chk("Constant", C(2.5), np.full(2, 2.5))
print("White(X,Y):", WhiteKernel(0.3)(a, b).ravel(), " White(X):",
      WhiteKernel(0.3)(a).tolist())
k = C(2.0) * RBF(1.5) + WhiteKernel(0.1)
print("theta is log of hyperparameters:", np.round(k.theta, 4), np.round(np.exp(k.theta), 4))
print("hyperparameter names:", [h.name for h in k.hyperparameters])
print("exponentiation:", np.allclose((RBF(1.0)**2)(a, b), RBF(1.0)(a, b)**2))
print("default GPR kernel:", GPR().fit(a, [0., 1.]).kernel_, "| alpha default",
      GPR().alpha)
# ---------- data
rng = np.random.default_rng(0)
X = np.sort(rng.uniform(0, 8, 15))[:, None]; y = np.sin(X[:, 0]) + 0.2*rng.normal(size=15)
Xs = np.linspace(-1, 9, 40)[:, None]
# alpha versus WhiteKernel
g1 = GPR(C(0.8, "fixed")*RBF(1.3, "fixed"), alpha=0.04, optimizer=None).fit(X, y)
g2 = GPR(C(0.8, "fixed")*RBF(1.3, "fixed")+WhiteKernel(0.04, "fixed"), alpha=1e-10, optimizer=None).fit(X, y)
m1, s1 = g1.predict(Xs, return_std=True); m2, s2 = g2.predict(Xs, return_std=True)
print("alpha vs WhiteKernel: same mean", np.allclose(m1, m2, atol=1e-6),
      "| std differ by sqrt(var+0.04):", np.allclose(s2**2, s1**2 + 0.04, atol=1e-6))
print("  LML equal:", np.isclose(g1.log_marginal_likelihood_value_, g2.log_marginal_likelihood_value_))
# alpha array
g3 = GPR(C(0.8, "fixed")*RBF(1.3, "fixed"), alpha=np.full(15, 0.04), optimizer=None).fit(X, y)
print("alpha array equals scalar:", np.allclose(g3.predict(Xs), m1))
# normalize_y
y100 = y + 100
n0 = GPR(C(1.0)*RBF(1.0), optimizer=None, alpha=0.04).fit(X, y100)
n1 = GPR(C(1.0)*RBF(1.0), optimizer=None, alpha=0.04, normalize_y=True).fit(X, y100)
far = np.array([[30.0]])
print("far prediction, y~100: normalize_y False %.3f  True %.3f" % (n0.predict(far)[0], n1.predict(far)[0]),
      "| train mean %.3f" % y100.mean())
print("normalize_y stores y_train_ mean/std:", round(n1.y_train_.mean(), 6), round(n1.y_train_.std(), 6))
OUT["normalize"] = [float(n0.predict(far)[0]), float(n1.predict(far)[0]), float(y100.mean())]
# unfitted prior
u = GPR(C(2.0)*RBF(1.0))
mu, su = u.predict(Xs[:3], return_std=True); print("unfitted prior mean/std:", mu, su)
# two targets share the kernel; hand-check values of Example 30.6-1 at x* = 0
Y2 = np.c_[y, 2*y + 1]; kf = C(0.8, "fixed")*RBF(1.3, "fixed")
g2t = GPR(kf, alpha=0.04, optimizer=None).fit(X, Y2)
sep = [GPR(kf, alpha=0.04, optimizer=None).fit(X, Y2[:, j]).predict(Xs) for j in range(2)]
print("two targets: predict shape", g2t.predict(Xs).shape, "| equals separate fits:", np.allclose(g2t.predict(Xs), np.c_[sep[0], sep[1]]))
print("RBF with one length-scale per feature:", RBF([1.0, 2.0])(np.array([[0., 0.], [1., 1.]]))[0, 1], "expected", np.exp(-0.5*(1/1 + 1/4)))
x2 = np.array([0., 2.]); y2_ = np.array([1.2, 0.4]); S2 = np.exp(-(x2[:, None]-x2[None, :])**2/2) + 0.1*np.eye(2); k0 = np.exp(-(x2-0.)**2/2)
print("Example 30.6-1 at x*=0: mean %.4f  var %.4f" % (k0 @ np.linalg.solve(S2, y2_), 1 - k0 @ np.linalg.solve(S2, k0)))
# gradient formula and lml(theta)
gp = GPR(C(1.0)*RBF(1.0), alpha=0.04).fit(X, y)
th = gp.kernel_.theta + np.array([0.4, -0.3]); lml, grad = gp.log_marginal_likelihood(th, eval_gradient=True)
num = []
for j in range(len(th)):
    e = np.zeros_like(th); e[j] = 1e-6
    num.append((gp.log_marginal_likelihood(th+e)-gp.log_marginal_likelihood(th-e))/2e-6)
print("LML gradient analytic", np.round(grad, 5), "numeric", np.round(num, 5))
# DotProduct = Bayesian linear regression = ridge on augmented columns
s0 = 0.9; sn2 = 0.3
gd = GPR(DotProduct(s0), alpha=sn2, optimizer=None).fit(X, y)
Xa = np.c_[X, np.full(15, s0)]
rg = Ridge(alpha=sn2, fit_intercept=False).fit(Xa, y)
print("DotProduct GP = ridge on [x, sigma_0]:", np.allclose(gd.predict(Xs), rg.predict(np.c_[Xs, np.full(40, s0)])))
# KRR equivalence, as stated in Chapter 21
ell = 1.3; kr = KernelRidge(alpha=0.04, kernel="rbf", gamma=1/(2*ell**2)).fit(X, y)
gr = GPR(RBF(ell, "fixed"), alpha=0.04, optimizer=None).fit(X, y)
print("GP mean = KernelRidge (gamma = 1/(2 l^2)):", np.allclose(gr.predict(Xs), kr.predict(Xs)),
      "| dual_coef_ = alpha_:", np.allclose(kr.dual_coef_, gr.alpha_))
# weight space and function space agree (Eq. 30.12-1)
Phi = np.random.default_rng(2).normal(size=(9, 4)); ph = np.random.default_rng(3).normal(size=4); s2 = 0.3
wsp = s2 * ph @ np.linalg.solve(Phi.T @ Phi + s2*np.eye(4), ph)
fsp = ph @ ph - (Phi @ ph) @ np.linalg.solve(Phi @ Phi.T + s2*np.eye(9), Phi @ ph)
mw = ph @ np.linalg.solve(Phi.T @ Phi + s2*np.eye(4), Phi.T @ np.arange(9.0))
mf = (Phi @ ph) @ np.linalg.solve(Phi @ Phi.T + s2*np.eye(9), np.arange(9.0))
print("weight-space variance = function-space variance:", np.isclose(wsp, fsp), "| means:", np.isclose(mw, mf))
# posterior variance does not depend on y
gA = GPR(C(0.8, "fixed")*RBF(1.3, "fixed"), alpha=0.04, optimizer=None)
sA = gA.fit(X, y).predict(Xs, return_std=True)[1]; sB = gA.fit(X, -3*y+5).predict(Xs, return_std=True)[1]
print("posterior std independent of y:", np.allclose(sA, sB))
# GPC facts
Xc = np.array([[-3.], [-2], [-1], [0.5], [1], [2], [3]]); yc = np.array([0, 0, 0, 1, 0, 1, 1])
c0 = GPC().fit(Xc, yc); print("GPC default kernel_:", c0.kernel_, "| fixed (no optimisation):", c0.kernel_.n_dims == 0 if hasattr(c0.kernel_, "n_dims") else None)
c1 = GPC(1.0*RBF(1.0)).fit(Xc, yc); print("GPC optimised kernel_:", c1.kernel_, "LML %.4f" % c1.log_marginal_likelihood_value_)
print("GPC defaults:", GPC().max_iter_predict, GPC().multi_class, GPC().warm_start)
Xi = np.random.default_rng(1).normal(size=(60, 2)); yi = (Xi[:, 0] > 0).astype(int) + (Xi[:, 1] > 0.5)
for mc in ("one_vs_rest", "one_vs_one"):
    try:
        cm = GPC(multi_class=mc).fit(Xi, yi); cm.predict_proba(Xi[:2]); print(mc, "predict_proba ok; kernel_ type", type(cm.kernel_).__name__, "| sum", cm.predict_proba(Xi[:2]).sum(1))
    except Exception as e: print(mc, "->", type(e).__name__, str(e)[:90])
# five-erf approximation of the sigmoid integral versus quadrature
bp = c1.base_estimator_; lm, lv = bp.latent_mean_and_variance(np.array([[-0.5], [0.0], [1.5], [6.0]]))
quad_p = [quad(lambda z: expit(z)*norm.pdf(z, m, np.sqrt(v)), -40, 40)[0] for m, v in zip(lm, lv)]
print("latent mean", np.round(lm, 4), "var", np.round(lv, 4))
print("predict_proba[:,1]", np.round(c1.predict_proba(np.array([[-0.5], [0.0], [1.5], [6.0]]))[:, 1], 5), "quadrature", np.round(quad_p, 5))
# timings
for n in (250, 500, 1000, 2000):
    Xn = np.random.default_rng(0).uniform(0, 10, (n, 1)); yn = np.sin(Xn[:, 0])
    dts = []
    for _ in range(5):
        t = time.perf_counter(); GPR(RBF(1.0), alpha=0.01, optimizer=None).fit(Xn, yn); dts.append(time.perf_counter()-t)
    dt = min(dts)
    print(f"n = {n:5d}: fit with fixed kernel {dt:.3f} s, K needs {8*n*n/1e6:.1f} MB"); OUT[f"t{n}"] = dt
print("n = 20000: K needs %.1f GB" % (8*20000**2/1e9))
json.dump(OUT, open("../verify/checks.json", "w"))
