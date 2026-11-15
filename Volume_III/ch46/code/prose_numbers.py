# Numbers quoted in the prose of Chapter 46 (and data for figures).
import json
import numpy as np
from scipy.stats import chi2
from sklearn.datasets import load_wine
from sklearn.covariance import (EmpiricalCovariance, LedoitWolf, OAS,
    MinCovDet, GraphicalLasso, GraphicalLassoCV, empirical_covariance,
    log_likelihood, ShrunkCovariance)
from sklearn.covariance._graph_lasso import alpha_max
from sklearn.preprocessing import StandardScaler
from returns import true_model, draw
Sigma, Theta, sd = true_model(); p = 20
N = {}
# --- spectrum, n = 30
X = draw(30, 0); S = empirical_covariance(X)
te, se = np.linalg.eigvalsh(Sigma), np.linalg.eigvalsh(S)
lw = LedoitWolf().fit(X); le = np.linalg.eigvalsh(lw.covariance_)
N["spectrum"] = dict(true=[te[0], te[-1]], sample=[se[0], se[-1]],
    lw=[le[0], le[-1]], cond_true=te[-1]/te[0], cond_sample=se[-1]/se[0],
    cond_lw=le[-1]/le[0], lw_alpha=lw.shrinkage_)
print("spectrum", {k: np.round(v, 4) for k, v in N["spectrum"].items()})
rng = np.random.RandomState(0)
ev = np.linalg.eigvalsh(np.cov(rng.normal(size=(3000, 2000)).T, bias=True))
print("Marchenko-Pastur p/n=2/3: simulated", ev.min().round(4), ev.max().round(4),
      " edges", ((1-np.sqrt(2/3))**2).round(4), ((1+np.sqrt(2/3))**2).round(4))
for n in (15, 20, 21):
    Xn = draw(n, 0); e = EmpiricalCovariance().fit(Xn)
    print("n", n, "rank", np.linalg.matrix_rank(e.covariance_),
          "score on 500 new days", round(e.score(draw(500, 99)), 2))
# --- score and mahalanobis
e = EmpiricalCovariance().fit(X); Xt = draw(200, 5)
Ste = empirical_covariance(Xt - e.location_, assume_centered=True)
print("score = formula:", np.isclose(e.score(Xt), 0.5*(np.linalg.slogdet(e.precision_)[1]
      - np.sum(Ste*e.precision_) - p*np.log(2*np.pi))))
d = Xt - e.location_
print("mahalanobis squared:", np.allclose(e.mahalanobis(Xt), np.einsum('ij,jk,ik->i', d, e.precision_, d)))
# --- partial correlation from the regression residuals (true Sigma)
i, j = 2, 3; rest = [k for k in range(p) if k not in (i, j)]
B = Sigma[np.ix_([i, j], rest)] @ np.linalg.inv(Sigma[np.ix_(rest, rest)])
R = Sigma[np.ix_([i, j], [i, j])] - B @ Sigma[np.ix_(rest, [i, j])]
print("partial corr from residuals", (R[0,1]/np.sqrt(R[0,0]*R[1,1])).round(4),
      " from precision", (-Theta[i,j]/np.sqrt(Theta[i,i]*Theta[j,j])).round(4),
      " plain corr", (Sigma[i,j]/np.sqrt(Sigma[i,i]*Sigma[j,j])).round(4))
# --- bias-variance of the shrinkage intensity (n = 30, 300 draws)
alphas = np.linspace(0, 1, 41); Rr = 300
acc = np.zeros((41, p, p)); mse = np.zeros(41); lws = []; oas = []; ld = []
for r in range(Rr):
    Xr = draw(30, 1000 + r); Sr = empirical_covariance(Xr); mu = np.trace(Sr)/p
    lws.append(LedoitWolf().fit(Xr).shrinkage_); oas.append(OAS().fit(Xr).shrinkage_)
    for k, a in enumerate(alphas):
        Sa = (1-a)*Sr + a*mu*np.eye(p); acc[k] += Sa; mse[k] += ((Sa-Sigma)**2).sum()/p
acc /= Rr; mse /= Rr
bias2 = np.array([((acc[k]-Sigma)**2).sum()/p for k in range(41)])
k0 = int(np.argmin(mse))
N["bv"] = dict(alphas=alphas.tolist(), mse=mse.tolist(), bias2=bias2.tolist(),
    var=(mse-bias2).tolist(), best=float(alphas[k0]), lw=float(np.mean(lws)),
    lw_sd=float(np.std(lws)), oas=float(np.mean(oas)), oas_sd=float(np.std(oas)))
print("shrinkage risk: alpha=0", mse[0].round(2), " best", alphas[k0], mse[k0].round(2),
      " alpha=1", mse[-1].round(2), " bias2/var at best", bias2[k0].round(2), (mse-bias2)[k0].round(2))
print("mean LW alpha", np.mean(lws).round(3), np.std(lws).round(3), " OAS", np.mean(oas).round(3))
# --- block_size, assume_centered, store_precision, error_norm
print("block_size irrelevant:", np.isclose(LedoitWolf(block_size=3).fit(X).shrinkage_, lw.shrinkage_))
ac = EmpiricalCovariance(assume_centered=True).fit(X)
print("assume_centered: location", ac.location_[:2], " cov = X'X/n:", np.allclose(ac.covariance_, X.T@X/30))
sp = EmpiricalCovariance(store_precision=False).fit(X)
print("store_precision=False: precision_", sp.precision_, " get_precision ok", np.allclose(sp.get_precision(), e.precision_))
print("error_norm: default", e.error_norm(Sigma).round(3), " no scaling", e.error_norm(Sigma, scaling=False).round(3),
      " not squared", e.error_norm(Sigma, squared=False).round(3), " spectral", e.error_norm(Sigma, norm='spectral').round(3))
# --- graphical lasso identities (standardised, n = 150)
Z = StandardScaler().fit_transform(draw(150, 1)); Sz = empirical_covariance(Z); a = 0.2
g = GraphicalLasso(alpha=a, enet_tol=1e-6, tol=1e-6).fit(Z); T, W = g.precision_, g.covariance_
off = lambda M: np.abs(M).sum() - np.abs(np.diag(M)).sum()
obj = -np.linalg.slogdet(T)[1] + np.sum(Sz*T) + a*off(T)
print("costs_ minus objective", (g.costs_[-1][0]-obj).round(4), " 2 p ln(2 pi) =", (2*p*np.log(2*np.pi)).round(4))
Rm = W - Sz; np.fill_diagonal(Rm, 0); nz = (np.abs(T) > 1e-10) & ~np.eye(p, dtype=bool)
print("KKT: W_ii = S_ii", np.abs(np.diag(W)-np.diag(Sz)).max(), " max|W-S| off", np.abs(Rm).max().round(6),
      " W-S = alpha sign(T) on edges", np.abs(Rm[nz]-a*np.sign(T[nz])).max().round(6), " symmetric", np.abs(T-T.T).max())
am = alpha_max(Sz)
print("alpha_max", am.round(4), " above:", (np.abs(GraphicalLasso(alpha=am*1.001).fit(Z).precision_) > 1e-10).sum()-p, "off-diagonal entries",
      " below:", ((np.abs(GraphicalLasso(alpha=am*0.99, enet_tol=1e-6).fit(Z).precision_) > 1e-10).sum()-p)//2, "edge")
print("lars vs cd", np.abs(GraphicalLasso(alpha=a, mode='lars', tol=1e-6).fit(Z).precision_ - T).max())
print("alpha=0 is the inverse of S:", np.allclose(GraphicalLasso(alpha=0.0).fit(Z).precision_, np.linalg.inv(Sz)))
cv = GraphicalLassoCV(enet_tol=1e-6).fit(Z); r = cv.cv_results_
o = np.argsort(r["alphas"])
N["glcv"] = dict(alphas=r["alphas"][o].tolist(), mean=r["mean_test_score"][o].tolist(), std=r["std_test_score"][o].tolist(), best=float(cv.alpha_), am=float(am))
print("CV alpha", cv.alpha_.round(4), "alpha_max", am.round(3), " ratio", (cv.alpha_/am).round(3), " grid", len(o), " smallest positive", r["alphas"][o][1].round(5))
# --- MCD: consistency factors, C-step monotonicity, affine invariance
c = lambda pp, aa: aa/chi2.cdf(chi2.ppf(aa, pp), pp+2)
print("c(0.8, p=2) =", c(2, 0.8).round(4), " c(0.5,p=4)", c(4, 0.5).round(4), " c(0.975,p=4)", c(4, 0.975).round(4), " c(0.975,p=2)", c(2, .975).round(4))
rs = np.random.RandomState(0); worst = 0
for t in range(200):
    Xc = rs.normal(size=(40, 3)) @ rs.normal(size=(3, 3)); h = 25; H = rs.permutation(40)[:h]; dets = []
    for it in range(4):
        m = Xc[H].mean(0); Sc = np.cov(Xc[H].T, bias=True); dets.append(np.linalg.det(Sc))
        dd = np.einsum('ij,jk,ik->i', Xc-m, np.linalg.inv(Sc), Xc-m); H = np.argsort(dd)[:h]
    worst = max(worst, max(np.diff(dets)))
print("C-step: largest increase of det over 800 steps", worst)
wine = load_wine(); cols = [0, 6, 9, 12]; Xw, yw = wine.data[:, cols], wine.target
rng = np.random.RandomState(0); bad = rng.choice(np.where(yw == 2)[0], 12, replace=False)
Wd = np.vstack([Xw[yw == 0], Xw[bad]]); isbad = np.r_[np.zeros(59, bool), np.ones(12, bool)]
cut = chi2(4).isf(0.025)
f1 = MinCovDet(random_state=0).fit(Wd).dist_ > cut
f2 = MinCovDet(random_state=0).fit(Wd * np.array([1000, .01, 5, .001])).dist_ > cut
print("affine invariance of the flags:", (f1 == f2).all(), " h =", int(np.ceil((71+4+1)/2)))
ref = EmpiricalCovariance().fit(Wd[~isbad]); sw = []
for sf in np.round(np.linspace(0.5, 1.0, 11), 2):
    m = MinCovDet(support_fraction=float(sf), random_state=0).fit(Wd)
    fl = m.dist_ > cut
    err = np.linalg.norm(m.covariance_-ref.covariance_)/np.linalg.norm(ref.covariance_)
    sw.append((float(sf), float(err), int(fl[isbad].sum()), int(fl[~isbad].sum()), int(m.raw_support_.sum())))
    print(f"support_fraction {sf:.2f}: relative error {err:.3f}, caught {fl[isbad].sum()}, false {fl[~isbad].sum()}, raw support {m.raw_support_.sum()}")
N["sf"] = sw
# 13 features
W13 = np.vstack([wine.data[yw == 0], wine.data[bad]]); m13 = MinCovDet(random_state=0).fit(W13); f13 = m13.dist_ > chi2(13).isf(0.025)
print("all 13 features: caught", f13[isbad].sum(), " false alarms", f13[~isbad].sum(), "of 59")
# empirical distance bound
print("max possible empirical d^2 for n = 71:", round((71-1)**2/71, 2))
json.dump(N, open("../verify/prose_numbers.json", "w"))
