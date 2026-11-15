# Numbers quoted in the text of Chapter 40 (not shown as a listing).
import numpy as np
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

x = np.array([154., 160., 164., 170., 176., 184.])
pi, mu, var = np.array([.5, .5]), np.array([158., 178.]), np.array([64., 64.])
d = pi * norm.pdf(x[:, None], mu, np.sqrt(var))          # pi_k N(x_i | k)
gam = d / d.sum(1, keepdims=True)

print("== 40.4: the bound for sample 1 (154 cm) at the starting values")
lnp = np.log(d[0].sum())
q = np.array([.5, .5])
F = np.sum(q * (np.log(d[0]) - np.log(q)))
kl = np.sum(q * np.log(q / gam[0]))
print("ln p(x1)", lnp.round(4), " bound F(q)", F.round(4),
      " gap", (lnp - F).round(4), " KL(q||gamma)", kl.round(4))
print("with q = gamma: bound", np.sum(gam[0] * (np.log(d[0]) - np.log(gam[0]))).round(4))
print("Jensen: ln of average", np.log(0.5 * 1 + 0.5 * 4).round(4),
      " average of ln", (0.5 * np.log(1) + 0.5 * np.log(4)).round(4))

print("== 40.5: responsibility of group 1 at 167 cm, centres 160 and 178")
for sd in (12, 6, 3, 2, 1):
    z = ((167 - 178) ** 2 - (167 - 160) ** 2) / (2 * sd ** 2)
    print(f"sigma {sd:2d}  z {z:7.4f}  gamma1 {1 / (1 + np.exp(-z)):.6f}")

print("== 40.8: the converged toy fit, evaluated at 170 cm")
gm = GaussianMixture(2, tol=1e-8, max_iter=500, weights_init=[.5, .5],
                     means_init=[[158.], [178.]],
                     precisions_init=[[[1 / 64]], [[1 / 64]]]).fit(x[:, None])
w, m, v = gm.weights_, gm.means_.ravel(), gm.covariances_.ravel()
print("pi", w.round(4), "mu", m.round(3), "var", v.round(3))
Nk = norm.pdf(170, m, np.sqrt(v)); num = w * Nk
print("N(170|k)", Nk.round(5), " pi_k N", num.round(5), " sum", num.sum().round(5))
print("gamma", (num / num.sum()).round(4), " ln p", np.log(num.sum()).round(4))
print("z-scores (170-mu)/sd", ((170 - m) / np.sqrt(v)).round(3))
print("mean ln L of the six heights", round(gm.score(x[:, None]), 4),
      " ln L total", round(gm.score(x[:, None]) * 6, 4))
one = GaussianMixture(1).fit(x[:, None])
print("one component: mean", one.means_.ravel().round(3), "var", one.covariances_.ravel().round(3),
      " ln L total", round(one.score(x[:, None]) * 6, 4))

print("== 40.3: the likelihood is unbounded (component 2 sits on 170 cm)")
for sd in (4., 1., 0.1, 0.01, 0.001):
    p = np.array([.5, .5]); mm = np.array([160., 170.]); vv = np.array([100., sd ** 2])
    dd = p * norm.pdf(x[:, None], mm, np.sqrt(vv))
    print(f"sigma_2 {sd:6.3f}   ln L {np.log(dd.sum(1)).sum():9.3f}")

print("== 40.11: lower_bound_ and score(X) (people data, default tol)")
from people import people_2d
Xp, _ = people_2d()
gp = GaussianMixture(3, random_state=0).fit(Xp)
print("lower_bound_", round(gp.lower_bound_, 6), " score(X)", round(gp.score(Xp), 6),
      " n_iter_", gp.n_iter_, " len(lower_bounds_)", len(gp.lower_bounds_),
      " last change", round(gp.lower_bounds_[-1] - gp.lower_bounds_[-2], 6))
g2 = GaussianMixture(3, random_state=0, tol=1e-8, max_iter=1000).fit(Xp)
print("tight tol:  lower_bound_", round(g2.lower_bound_, 6), " score(X)", round(g2.score(Xp), 6), " n_iter_", g2.n_iter_)

print("== 40.6 / 40.12: BIC on the six heights and on the 400 heights")
for k in (1, 2):
    gk = GaussianMixture(k, tol=1e-8, max_iter=500, random_state=0).fit(x[:, None])
    print("six heights  K =", k, " BIC", round(gk.bic(x[:, None]), 3))
from people import heights_1d
h, _ = heights_1d()
for k in (1, 2, 3, 4):
    gk = GaussianMixture(k, n_init=3, tol=1e-6, max_iter=500, random_state=0).fit(h[:, None])
    print("400 heights  K =", k, " BIC", round(gk.bic(h[:, None]), 1))

print("== 40.13: wine, diag, K = 3")
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
Xw, yw = load_wine(return_X_y=True); Xw = StandardScaler().fit_transform(Xw)
g3 = GaussianMixture(3, covariance_type="diag", n_init=5, tol=1e-5, max_iter=500, random_state=0).fit(Xw)
lab = g3.predict(Xw)
for c in range(3):
    print("cultivar", c, [int(np.sum((yw == c) & (lab == k))) for k in range(3)])
print("== 40.13: wine, diag, K = 4")
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
Xw, yw = load_wine(return_X_y=True); Xw = StandardScaler().fit_transform(Xw)
g4 = GaussianMixture(4, covariance_type="diag", n_init=5, tol=1e-5, max_iter=500, random_state=0).fit(Xw)
lab = g4.predict(Xw)
for c in range(3):
    print("cultivar", c, [int(np.sum((yw == c) & (lab == k))) for k in range(4)])

print("== 40.8: intermediate numbers of the posterior at 170 cm (Section 40.11)")
dd = 170 - m
print("deviation", dd.round(3), " squared", (dd ** 2).round(2), " /var", (dd ** 2 / v).round(4))
print("exp(-half)", np.exp(-0.5 * dd ** 2 / v).round(5), " 1/sqrt(2 pi var)", (1 / np.sqrt(2 * np.pi * v)).round(5))
