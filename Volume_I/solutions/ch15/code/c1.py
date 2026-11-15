# Problem 15C.1: 200 samples, 30 features, degree-3 polynomial columns.
import numpy as np
from math import comb
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler
np.set_printoptions(precision=4, suppress=True)
p, n = 30, 200
print("columns, degree 3:", comb(p+3, 3), PolynomialFeatures(3).fit(np.zeros((1, p))).n_output_features_,
      " without the bias:", comb(p+3, 3) - 1)
print("degree 2:", comb(p+2, 2), " degree 3 interaction_only:", sum(comb(p, k) for k in range(4)),
      " degree 2 interaction_only:", sum(comb(p, k) for k in range(3)),
      " cubic splines, 5 knots, one per feature:", p*(5+3-1))
print("ratio columns / samples:", round((comb(p+3, 3) - 1) / n, 1))

rng = np.random.default_rng(0)
def make(m):
    X = rng.normal(size=(m, p))
    return X, X[:, 0] + X[:, 1]**2 + X[:, 2]*X[:, 3] + 0.5*rng.normal(size=m)   # a few real terms, noise 0.5
Xtr, ytr = make(n); Xte, yte = make(2000)
pf = PolynomialFeatures(3, include_bias=False)
sc = StandardScaler().fit(Xtr)
Ftr = pf.fit_transform(sc.transform(Xtr)); Fte = pf.transform(sc.transform(Xte))
ss = StandardScaler().fit(Ftr); Ftr, Fte = ss.transform(Ftr), ss.transform(Fte)
print("rank of the centred column matrix:", np.linalg.matrix_rank(Ftr - Ftr.mean(0)), " (n - 1 =", n - 1, ")")
import warnings
with warnings.catch_warnings(record=True) as rec:
    warnings.simplefilter('always')
    ols = LinearRegression().fit(Ftr, ytr)
print('warnings raised by LinearRegression.fit:', len(rec))
print("LinearRegression  rank_", ols.rank_, " train R2 %.4f  test R2 %.4f  largest |weight| %.2e"
      % (ols.score(Ftr, ytr), ols.score(Fte, yte), np.abs(ols.coef_).max()))
mn = np.linalg.pinv(np.c_[np.ones(n), Ftr]) @ ytr
print("minimum-norm solution of pinv agrees with lstsq:", np.allclose(mn[1:], ols.coef_, atol=1e-6))
rcv = RidgeCV(alphas=np.logspace(-2, 5, 22)).fit(Ftr, ytr)
print("RidgeCV  alpha %.3g  train R2 %.4f  test R2 %.4f" % (rcv.alpha_, rcv.score(Ftr, ytr), rcv.score(Fte, yte)))
las = LassoCV(cv=5, random_state=0, max_iter=20000).fit(Ftr, ytr)
print("LassoCV  alpha %.4f  non-zero weights %d  train R2 %.4f  test R2 %.4f"
      % (las.alpha_, (las.coef_ != 0).sum(), las.score(Ftr, ytr), las.score(Fte, yte)))
lin = LinearRegression().fit(sc.transform(Xtr), ytr)
print("plain linear model on the 30 features: test R2 %.4f" % lin.score(sc.transform(Xte), yte))
