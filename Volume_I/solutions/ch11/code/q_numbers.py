# Numbers quoted in the model answers to the questions for discussion of Chapter 11
import time
import numpy as np
from sklearn.datasets import load_diabetes, make_regression
from sklearn.linear_model import (Lasso, LassoCV, LinearRegression, Ridge, lars_path,
                                  LassoLarsCV, LassoLarsIC, lasso_path)
from sklearn.model_selection import train_test_split, cross_val_score
np.set_printoptions(precision=4, suppress=True)

print("== Q1  how often does a lasso solution have an exact zero as p grows?")
rng = np.random.default_rng(0)
def shrink_l1(w, t):                      # minimise |w - what|^2 subject to |w|_1 <= t (orthonormal design)
    a = np.sort(np.abs(w))[::-1]; cs = np.cumsum(a)
    k = np.arange(1, len(a) + 1); th = (cs - t) / k
    kk = np.flatnonzero(a > th)[-1]; return np.sign(w) * np.maximum(np.abs(w) - th[kk], 0)
for p in (2, 3, 5, 10, 20, 50, 200):
    z, allnz = [], 0
    for _ in range(4000):
        wh = rng.standard_normal(p); w = shrink_l1(wh, 0.5 * np.abs(wh).sum())
        z.append((w == 0).mean()); allnz += (w != 0).all()
    print(f"p = {p:3d}: corners 2p = {2*p:4d}  share of zero weights = {np.mean(z):.3f}   P(no weight is zero) = {allnz/4000:.3f}")

print("\n== Q2  one feature, v = 1: lasso S(c, 1)/v against ridge c/(v + lam) with lam = 0.5")
soft = lambda z, a: float(np.sign(z) * max(abs(z) - a, 0.0))
for c in (0.6, 3.0, 30.0):
    print(f"c = {c:5.1f}: least squares {c:6.2f}  lasso {soft(c, 1.0):6.2f}  ridge {c / 1.5:6.2f}   lasso loss of weight {c - soft(c, 1.0):5.2f}, ridge loss {c - c / 1.5:6.2f}")
print("ridge never zero: c = 0.01 gives", 0.01 / 1.5, "; lasso gives", soft(0.01, 1.0))

print("\n== Q3  diabetes: does the lasso lose anything against least squares?")
d = load_diabetes(); X, y = d.data, d.target
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
ols = LinearRegression().fit(Xtr, ytr); las = LassoCV(cv=5, random_state=0).fit(Xtr, ytr)
print("test R2: OLS %.3f  LassoCV %.3f (%d features)   RMSE %.2f vs %.2f" % (
    ols.score(Xte, yte), las.score(Xte, yte), (las.coef_ != 0).sum(),
    np.sqrt(((ols.predict(Xte) - yte) ** 2).mean()), np.sqrt(((las.predict(Xte) - yte) ** 2).mean())))
print("nested CV R2: OLS %.3f  LassoCV %.3f" % (cross_val_score(LinearRegression(), X, y, cv=5).mean(),
      cross_val_score(LassoCV(cv=5, random_state=0), X, y, cv=5).mean()))
print("s1, s2 weights: OLS", ols.coef_[4:6].round(1), " lasso", (las.coef_[4:6].round(1) + 0.0))
al, co, _ = lasso_path(Xtr - Xtr.mean(0), ytr - ytr.mean(), alphas=100)
best = {}
for j in range(co.shape[1]):
    k = int((co[:, j] != 0).sum()); m = Lasso(alpha=al[j]).fit(Xtr, ytr); best[k] = max(best.get(k, -9), m.score(Xte, yte))
print("best test R2 on the path by number of features:", {k: round(v, 3) + 0.0 for k, v in sorted(best.items())})

print("\n== Q4  one LARS path against five-fold cross-validation (data of Listing 11.4)")
X4, y4 = make_regression(n_samples=120, n_features=20, n_informative=4, noise=25.0, random_state=2)
a_full, _, _ = lars_path(X4, y4, method="lasso"); print("nodes of the path on all 120 samples:", len(a_full))
cv = LassoLarsCV(cv=5).fit(X4, y4)
print("LassoLarsCV: alpha_ = %.4f ; mse_path_ shape %s ; folds %d ; path computations = 5 fold paths + 1 final fit = 6" % (cv.alpha_, cv.mse_path_.shape, 5))
ic = LassoLarsIC(criterion="bic").fit(X4, y4); print("LassoLarsIC: alpha_ = %.4f ; criterion_ values %d ; path computations = 1" % (ic.alpha_, len(ic.criterion_)))
def timeit(f, r=15):
    t = []
    for _ in range(r):
        s = time.perf_counter(); f(); t.append(time.perf_counter() - s)
    return np.median(t)
tcv = timeit(lambda: LassoLarsCV(cv=5).fit(X4, y4)); tic = timeit(lambda: LassoLarsIC(criterion="bic").fit(X4, y4))
print("median fit time: LassoLarsCV %.1f ms, LassoLarsIC %.1f ms, ratio %.1f" % (1e3 * tcv, 1e3 * tic, tcv / tic))
