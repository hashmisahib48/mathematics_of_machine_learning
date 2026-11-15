# Checks of statements made in the text of Chapter 35.
import warnings
import numpy as np
from scipy.linalg import svd
from sklearn.cross_decomposition import (CCA, PLSCanonical,
                                         PLSRegression, PLSSVD)
from sklearn.datasets import load_linnerud
from sklearn.linear_model import LinearRegression
from spectra import make_spectra

Xc = np.array([[-3., 1], [-2, -1], [2, 1], [3, -1]])
Y2 = np.array([[0., 1], [-1, 0], [2, -1], [-1, 0]])
y1 = Y2[:, 0]
np.set_printoptions(precision=4, suppress=True)

p = PLSRegression(1, scale=False).fit(Xc, Y2)
print("1. NIPALS of nipals_toy.py: n_iter_ =", p.n_iter_,
      " x_weights_ =", p.x_weights_[:, 0])
U, s, Vt = svd(Xc.T @ Y2)
print("   SVD of X^T Y: u1 =", U[:, 0], " sigma =", s)

rng = np.random.default_rng(0)
X = rng.standard_normal((50, 5)) @ rng.standard_normal((5, 5))
y = X @ [1, -2, 0.5, 0, 3] + rng.standard_normal(50)
for A in (4, 5):
    d = abs(PLSRegression(A, scale=False).fit(X, y).coef_
            - LinearRegression().fit(X, y).coef_).max()
    print(f"2. n = 50, p = 5, A = {A}: max |coef_ PLS - OLS| = {d:.1e}")

w = X.T @ (y - y.mean())
w /= np.linalg.norm(w)
Xm, ym = X - X.mean(0), y - y.mean()
dirs = rng.standard_normal((100000, 5))
dirs /= np.linalg.norm(dirs, axis=1)[:, None]
print("3. covariance of the score with y: Xty direction",
      f"{(Xm @ w) @ ym:.4f}, largest of 100000 random",
      f"{((Xm @ dirs.T).T @ ym).max():.4f}, norm of X^T y",
      f"{np.linalg.norm(Xm.T @ ym):.4f}")

pl = PLSRegression(2).fit(X, y)
print("4. 1-D target: coef_ shape", pl.coef_.shape, " predict shape",
      pl.predict(X).shape, " x_std ddof=1:",
      np.allclose(pl._x_std, X.std(0, ddof=1)))
Z = (X - X.mean(0)) / X.std(0, ddof=1)
print("   transform uses x_rotations_: ",
      abs(pl.transform(X) - Z @ pl.x_rotations_).max() < 1e-12,
      "; x_weights_ would differ by",
      f"{abs(pl.transform(X) - Z @ pl.x_weights_).max():.3f}")

T = pl.x_scores_
print("5. scores orthogonal: t1.t2 =", f"{T[:, 0] @ T[:, 1]:.1e}",
      " prediction = T q^T:",
      np.allclose(pl.predict(X), (T @ pl.y_loadings_.T).ravel()
                  * y.std(ddof=1) + y.mean()))

try:
    PLSCanonical(2).fit(X, y)
except ValueError as e:
    print("6. PLSCanonical, one target:", e)
print("   PLSSVD has predict:", hasattr(PLSSVD(), "predict"),
      " transform:", hasattr(PLSSVD(), "transform"))

lin = load_linnerud()
L, M = lin.data, lin.target
z = lambda a: (a - a.mean(0)) / a.std(0, ddof=1)
Ls, Ms = z(L), z(M)
Rxx, Ryy, Rxy = Ls.T @ Ls, Ms.T @ Ms, Ls.T @ Ms
ev = np.linalg.eigvals(np.linalg.solve(Rxx, Rxy @ np.linalg.solve(Ryy, Rxy.T)))
print("7. canonical correlations, sqrt of eigenvalues:",
      np.sqrt(np.sort(ev.real)[::-1]))
cca = CCA(3).fit(L, M)
a, b = cca.transform(L, M)
print("   CCA scores correlation:",
      [round(float(np.corrcoef(a[:, k], b[:, k])[0, 1]), 4)
       for k in range(3)])
sv = PLSSVD(3).fit(L, M)
Us, ss, _ = svd(Rxy)
print("   PLSSVD x_weights_ = SVD of X^T Y:",
      np.allclose(abs(sv.x_weights_), abs(Us)), " singular values/(n-1):",
      ss / (len(L) - 1))
pc = PLSCanonical(3).fit(L, M)
print("   PLSCanonical vs PLSSVD x weights, cosine per component:",
      [round(float(abs(pc.x_weights_[:, k] @ sv.x_weights_[:, k])), 4)
       for k in range(3)])
pr = PLSRegression(3).fit(L, M)
print("   PLSRegression vs PLSSVD x weights, cosine per component:",
      [round(float(abs(pr.x_weights_[:, k] @ sv.x_weights_[:, k])), 4)
       for k in range(3)])

def first_corr(A_, B_):
    c = CCA(1).fit(A_, B_)
    s_, u_ = c.transform(A_, B_)
    return np.corrcoef(s_[:, 0], u_[:, 0])[0, 1]

obs = first_corr(L, M)
perm = np.array([first_corr(L, M[rng.permutation(20)])
                 for _ in range(500)])
print("8. first canonical correlation", f"{obs:.3f}; 500 row permutations",
      f"of Y: mean {perm.mean():.3f}, 95th percentile",
      f"{np.quantile(perm, .95):.3f}; {int((perm >= obs).sum())}",
      "of 500 reach it")
from sklearn.model_selection import KFold
oa, ob = np.zeros(20), np.zeros(20)
for tr, te in KFold(5, shuffle=True, random_state=0).split(L):
    c = CCA(1).fit(L[tr], M[tr])
    s_, u_ = c.transform(L[te], M[te])
    oa[te], ob[te] = s_[:, 0], u_[:, 0]
print("   held-out first-pair correlation (5 folds, pooled):",
      f"{np.corrcoef(oa, ob)[0, 1]:.3f}")

Xs, ys = make_spectra(60, 0)
with warnings.catch_warnings(record=True) as rec:
    warnings.simplefilter("always")
    try:
        PLSRegression(60).fit(Xs, ys)
        print("9. n_components = 60 (n = 60, p = 100): fitted;",
              [str(r.message) for r in rec][:1])
    except Exception as e:
        print("9. n_components = 60:", type(e).__name__, e)
try:
    PLSRegression(61).fit(Xs, ys)
except ValueError as e:
    print("   n_components = 61:", e)
print("   rank of centred spectra:",
      np.linalg.matrix_rank(Xs - Xs.mean(0)))

print("\n10. scale=False on the spectra (A = 6)")
from sklearn.model_selection import KFold, cross_val_score
Xt_, yt_ = make_spectra(1000, 1)
cvk = KFold(5, shuffle=True, random_state=0)
for sc_flag in (True, False):
    m = PLSRegression(6, scale=sc_flag)
    cvr = -cross_val_score(m, Xs, ys, cv=cvk,
                           scoring="neg_root_mean_squared_error").mean()
    te = float(np.sqrt(((m.fit(Xs, ys).predict(Xt_).ravel() - yt_) ** 2).mean()))
    print(f"    scale={sc_flag!s:5s}  CV RMSE {cvr:.2f}  test RMSE {te:.2f}")

print("11. first canonical correlation of pure noise, n = 25, L = 3")
def canon1(A_, B_):
    A_, B_ = A_ - A_.mean(0), B_ - B_.mean(0)
    Qa = np.linalg.svd(A_, full_matrices=False)
    Qb = np.linalg.svd(B_, full_matrices=False)
    ra, rb = (Qa[1] > 1e-10).sum(), (Qb[1] > 1e-10).sum()
    return np.linalg.svd(Qa[0][:, :ra].T @ Qb[0][:, :rb], compute_uv=False)[0]
rg = np.random.default_rng(1)
for pp in (3, 10, 20, 24, 30):
    vals = [canon1(rg.standard_normal((25, pp)), rg.standard_normal((25, 3)))
            for _ in range(200)]
    print(f"    p = {pp:2d}: mean {np.mean(vals):.3f}  min {np.min(vals):.3f}")

print("12. Linnerud, A = 1: one PLS2 against three PLS1 (5-fold R2)")
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_predict
P2 = cross_val_predict(PLSRegression(1), L, M, cv=cvk)
P1 = np.column_stack([cross_val_predict(PLSRegression(1), L, M[:, j], cv=cvk)
                      for j in range(3)])
print("    PLS2 :", r2_score(M, P2, multioutput="raw_values").round(3))
print("    PLS1 :", r2_score(M, P1, multioutput="raw_values").round(3))
print("13. predict exists:", {c.__name__: hasattr(c(), "predict")
      for c in (PLSRegression, PLSCanonical, CCA, PLSSVD)},
      " x_scores_ after fit:", hasattr(PLSCanonical().fit(L, M), "x_scores_"),
      hasattr(PLSRegression().fit(L, M), "x_scores_"))

pl4 = PLSRegression(4).fit(Xs, ys)
PW = pl4.x_loadings_.T @ pl4.x_weights_
print("14. P^T W for the spectra, A = 4 (upper triangular, unit diagonal):")
print(np.array2string(PW, precision=4, suppress_small=True))
print("    largest entry below the diagonal:",
      f"{abs(np.tril(PW, -1)).max():.1e}")

print("15. PLS1 and conjugate gradients (scaled spectra, A = 1..6)")
Zs = (Xs - Xs.mean(0)) / Xs.std(0, ddof=1)
ysc = (ys - ys.mean()) / ys.std(ddof=1)
G, g0 = Zs.T @ Zs, Zs.T @ ysc
b, r, d = np.zeros(100), g0.copy(), g0.copy()      # CG on G b = g0
for A_ in range(1, 7):
    Gd = G @ d
    alpha = (r @ r) / (d @ Gd)
    b = b + alpha * d
    r_new = r - alpha * Gd
    d = r_new + (r_new @ r_new) / (r @ r) * d
    r = r_new
    pls_b = (PLSRegression(A_).fit(Xs, ys).coef_.ravel()
             * Xs.std(0, ddof=1) / ys.std(ddof=1))
    print(f"    A = {A_}: max |B_PLS - B_CG| = {abs(pls_b - b).max():.1e}")
