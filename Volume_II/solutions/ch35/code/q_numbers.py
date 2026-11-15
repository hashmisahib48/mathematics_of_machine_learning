# Numbers quoted in the model answers to the questions of Chapter 35.
import sys, os, warnings
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../ch35/code'); os.chdir(HERE + '/..')
import numpy as np
from sklearn.cross_decomposition import PLSRegression, CCA
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from spectra import make_spectra
warnings.simplefilter('ignore', UserWarning)
np.set_printoptions(precision=3, suppress=True)
X, y = make_spectra(60, 0); Xt, yt = make_spectra(1000, 1)
sc = StandardScaler().fit(X); Z, Zt = sc.transform(X), sc.transform(Xt); yc = y - y.mean(); ytc = yt - y.mean()

def stats(w, name):                      # spread, correlation, covariance of the score; on the training data and on 1000 new samples
    t, tt = Z @ w, Zt @ w
    print(f"{name:14s} train: sd {t.std(ddof=1):6.3f} corr {np.corrcoef(t, y)[0, 1]:6.3f}   new: sd {tt.std(ddof=1):6.3f} corr {np.corrcoef(tt, yt)[0, 1]:6.3f}")
print("== Q1  first direction of three rules on the 60 spectra (scaled channels)")
pca_w = PCA(1).fit(Z).components_[0]; g = Z.T @ yc; pls_w = g / np.linalg.norm(g)
ols_w = np.linalg.pinv(Z) @ yc; ols_w /= np.linalg.norm(ols_w)
stats(pca_w, "PCA"); stats(pls_w, "PLS"); stats(ols_w, "least squares")
print("sd of y on the training data %.3f" % y.std(ddof=1))

print("\n== Q2  correlation of the k-th score with y, and the share of variance of Z in it")
P = PCA(15).fit(Z); Tp = P.transform(Z); pl = PLSRegression(15).fit(X, y); Tl = pl.x_scores_
print(" k  PCA: var share   corr^2 with y | PLS: corr^2 with y")
for k in range(15):
    print(f"{k + 1:2d}  {P.explained_variance_ratio_[k]:12.3f} {np.corrcoef(Tp[:, k], y)[0, 1] ** 2:13.3f} | {np.corrcoef(Tl[:, k], y)[0, 1] ** 2:15.3f}")
print("sum of corr^2 (training R^2) for k = 1..6, 7..15, PCA: %.3f %.3f   PLS: %.3f %.3f" % tuple(
    [sum(np.corrcoef(T_[:, k], y)[0, 1] ** 2 for k in rg) for T_ in (Tp, Tl) for rg in (range(6), range(6, 15))][i] for i in (0, 1, 2, 3)))
print("correlation of PCA scores 1..4 with y:", [round(float(np.corrcoef(Tp[:, k], y)[0, 1]), 3) for k in range(4)])
print("PLS scores 1..4:", [round(float(np.corrcoef(Tl[:, k], y)[0, 1]), 3) for k in range(4)])
print("cumulative explained variance of the first 4 PCA components of the scaled spectra: %.3f" % P.explained_variance_ratio_[:4].sum())

print("\n== Q3  cross-validated RMSE, 5 folds, for A = 5 ... 15")
cv = KFold(5, shuffle=True, random_state=0)
f = lambda m: -cross_val_score(m, X, y, cv=cv, scoring='neg_root_mean_squared_error').mean()
print(" A   PLS    PCA+OLS")
for A in range(5, 16, 2):
    print(f"{A:2d} {f(PLSRegression(A)):6.3f} {f(make_pipeline(StandardScaler(), PCA(A), LinearRegression())):8.3f}")

print("\n== Q4  CCA on pure noise, n = 25, three targets")
rng = np.random.default_rng(0)
for p in (3, 10, 20, 24, 30):
    r_in, r_new = [], []
    for _ in range(20):
        A_, B_ = rng.standard_normal((25, p)), rng.standard_normal((25, 3)); A2, B2 = rng.standard_normal((25, p)), rng.standard_normal((25, 3))
        m = CCA(1, max_iter=2000).fit(A_, B_); a, b = m.transform(A_, B_); r_in.append(np.corrcoef(a[:, 0], b[:, 0])[0, 1])
        a2, b2 = m.transform(A2, B2); r_new.append(np.corrcoef(a2[:, 0], b2[:, 0])[0, 1])
    print(f"p = {p:2d}: first canonical correlation in the sample {np.mean(r_in):.3f}, on 25 new samples {np.mean(r_new):+.3f} (mean of 20 draws)")
