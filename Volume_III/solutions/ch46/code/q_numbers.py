# Numbers quoted in the model answers to the questions for discussion (Q1 - Q4) of Chapter 46
import sys, numpy as np
sys.path.insert(0, '../../../ch46/code')
from returns import true_model, draw
from scipy.stats import chi2
from sklearn.datasets import load_wine
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import (EmpiricalCovariance, LedoitWolf, GraphicalLassoCV, MinCovDet, empirical_covariance, log_likelihood)
np.set_printoptions(precision=4, suppress=True)
Sigma, Theta, sd = true_model(); p = 20
true_edge = (np.abs(Theta) > 1e-12) & ~np.eye(p, dtype=bool)

print("== Q1  divisor n or n - 1")
X, Xte = draw(30, 0), draw(2000, 99); n = len(X)
S = empirical_covariance(X); Sb = S * n / (n - 1); ev = np.linalg.eigvalsh
print("factor n/(n-1): n = 5: %.4f, n = 30: %.4f, n = 480: %.4f" % (5 / 4, 30 / 29, 480 / 479))
Ste = empirical_covariance(Xte - X.mean(0), assume_centered=True)
print("test log-likelihood of the stocks, divisor n: %.2f, divisor n-1: %.2f" % (log_likelihood(Ste, np.linalg.inv(S)), log_likelihood(Ste, np.linalg.inv(Sb))))
e = lambda M: ((M - Sigma)**2).sum() / p
er = [(e(empirical_covariance(draw(30, s))), e(empirical_covariance(draw(30, s)) * 30 / 29)) for s in range(300)]
print("squared error against the truth (mean of 300 draws), divisor n: %.2f, divisor n-1: %.2f" % (np.mean([a for a, b in er]), np.mean([b for a, b in er])))
print("smallest eigenvalue of S: %.3f (n), %.3f (n-1); true %.3f;  largest %.2f / %.2f; true %.2f" % (ev(S)[0], ev(Sb)[0], ev(Sigma)[0], ev(S)[-1], ev(Sb)[-1], ev(Sigma)[-1]))
D = X - X.mean(0); d2n = np.einsum('ij,jk,ik->i', D, np.linalg.inv(S), D); d2b = np.einsum('ij,jk,ik->i', D, np.linalg.inv(Sb), D)
print("sum of the 30 squared distances: %.2f (divisor n; n p = %d), %.2f (divisor n-1); largest %.2f vs %.2f" % (d2n.sum(), n * p, d2b.sum(), d2n.max(), d2b.max()))
sl = LedoitWolf().fit(X); print("Ledoit-Wolf intensity %.3f; its error %.2f; error of the shrunk estimate from the n-1 matrix:" % (sl.shrinkage_, e(sl.covariance_)),
      "%.2f" % e((1 - sl.shrinkage_) * Sb + sl.shrinkage_ * np.trace(Sb) / p * np.eye(p)))

print("\n== Q2  why the largest eigenvalue is too large and the smallest too small")
rat = np.array([[ev(empirical_covariance(draw(30, s)))[-1] / ev(Sigma)[-1], ev(empirical_covariance(draw(30, s)))[0] / ev(Sigma)[0]] for s in range(300)])
print("stocks, 300 draws of 30 days: lambda_max(S)/lambda_1 mean %.3f (min %.3f, max %.3f);  lambda_min(S)/lambda_p mean %.3f (min %.3f, max %.3f)" % (rat[:, 0].mean(), rat[:, 0].min(), rat[:, 0].max(), rat[:, 1].mean(), rat[:, 1].min(), rat[:, 1].max()))
print("one draw (seed 0): lambda_max(S) %.1f, true %.1f; lambda_min(S) %.3f, true %.3f" % (ev(S)[-1], ev(Sigma)[-1], ev(S)[0], ev(Sigma)[0]))
print("fraction of draws with the largest too large: %.3f, with the smallest too small: %.3f" % ((rat[:, 0] > 1).mean(), (rat[:, 1] < 1).mean()))
r = np.random.RandomState(0); L = np.array([ev(empirical_covariance(r.normal(size=(30, 20)))) for _ in range(300)])
print("true covariance I (p = 20, n = 30): mean smallest %.3f, mean largest %.2f; Marchenko-Pastur edges (1 -+ sqrt(p/n))^2 = %.3f, %.2f" % (L[:, 0].mean(), L[:, -1].mean(), (1 - np.sqrt(20 / 30))**2, (1 + np.sqrt(20 / 30))**2))
print("trace: mean over the 300 draws of tr S / tr Sigma = %.3f ( (n-1)/n = %.3f )" % (np.mean([np.trace(empirical_covariance(draw(30, s))) for s in range(300)]) / np.trace(Sigma), 29 / 30))
v1 = np.linalg.eigh(Sigma)[1][:, -1]; print("Rayleigh quotient of the true top eigenvector, mean over draws: %.3f of lambda_1" % (np.mean([v1 @ empirical_covariance(draw(30, s)) @ v1 for s in range(300)]) / ev(Sigma)[-1]))

print("\n== Q3  55 edges against 18")
sc = StandardScaler().fit(X); Z = sc.transform(X)
gl = GraphicalLassoCV(enet_tol=1e-6).fit(Z); P = gl.precision_; E = (np.abs(P) > 1e-10) & ~np.eye(p, dtype=bool)
print("alpha_ %.3f, edges %d, true %d, false %d, missed %d" % (gl.alpha_, E.sum() // 2, (E & true_edge).sum() // 2, (E & ~true_edge).sum() // 2, (~E & true_edge).sum() // 2))
d = np.sqrt(np.diag(P)); pc = -P / np.outer(d, d); iu = np.triu_indices(p, 1); strength = np.abs(pc[iu]); te = true_edge[iu]; fe = E[iu] & ~te
print("partial correlations of the found edges: true edges mean %.3f, false edges mean %.3f (max %.3f)" % (strength[E[iu] & te].mean(), strength[fe].mean(), strength[fe].max()))
top = np.argsort(-np.where(E[iu], strength, 0))[:18]; print("of the 18 strongest found edges, %d are true" % te[top].sum())
r_ = gl.cv_results_; k = np.argmax(r_['mean_test_score']); thr = r_['mean_test_score'][k] - r_['std_test_score'][k]
a1 = max(a for a, m in zip(r_['alphas'], r_['mean_test_score']) if m >= thr)
from sklearn.covariance import graphical_lasso
_, P1 = graphical_lasso(empirical_covariance(Z), alpha=a1, enet_tol=1e-6); E1 = (np.abs(P1) > 1e-10) & ~np.eye(p, dtype=bool)
print("one-standard-error rule: alpha %.3f (against %.3f): edges %d, true %d, false %d" % (a1, gl.alpha_, E1.sum() // 2, (E1 & true_edge).sum() // 2, (E1 & ~true_edge).sum() // 2))
Z150 = StandardScaler().fit_transform(draw(150, 1)); g150 = GraphicalLassoCV(enet_tol=1e-6).fit(Z150); E150 = (np.abs(g150.precision_) > 1e-10) & ~np.eye(p, dtype=bool)
print("with 150 days: alpha_ %.3f, edges %d, true %d, false %d" % (g150.alpha_, E150.sum() // 2, (E150 & true_edge).sum() // 2, (E150 & ~true_edge).sum() // 2))

print("\n== Q4  intruders and clean wines")
wine = load_wine(); y = wine.target; W4 = wine.data[:, [0, 6, 9, 12]]; bad = np.random.RandomState(0).choice(np.where(y == 2)[0], 12, replace=False)
Wd = np.vstack([W4[y == 0], W4[bad]]); isb = np.r_[np.zeros(59, bool), np.ones(12, bool)]; cut = chi2(4).isf(0.025)
for nm, m in (("empirical", EmpiricalCovariance().fit(Wd)), ("MinCovDet", MinCovDet(random_state=0).fit(Wd))):
    d2 = m.mahalanobis(Wd); f = d2 > cut; tp, fp = f[isb].sum(), f[~isb].sum()
    print("%-10s caught %2d of 12, false alarms %d of 59: precision %.3f recall %.3f F1 %.3f; AUC of d^2 %.3f; largest d^2 of an intruder %.1f, of a clean wine %.1f" % (nm, tp, fp, tp / max(tp + fp, 1), tp / 12, 2 * tp / (tp + fp + 12), roc_auc_score(isb, d2), d2[isb].max(), d2[~isb].max()))
de = EmpiricalCovariance().fit(Wd).mahalanobis(Wd)
print("empirical covariance with a threshold set just above its largest clean distance (%.1f): intruders above it: %d of 12" % (de[~isb].max(), (de[isb] > de[~isb].max()).sum()))
print("expected false alarms at the 97.5%% point: %.2f of 59;  the MinCovDet detector is the better one if a false alarm costs less than (12-2)/(2-0) = %.0f times a miss" % (59 * 0.025, (12 - 2) / (2 - 0)))
