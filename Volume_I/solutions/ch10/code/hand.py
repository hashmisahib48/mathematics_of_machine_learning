# Hand checks for Problems 10A.1 - 10A.4 and 10C.1 (every number quoted in the solutions)
import numpy as np
from sklearn.linear_model import Ridge
np.set_printoptions(precision=4, suppress=True)

x = np.array([0, 1, 2, 3.]); y = np.array([1, 3, 4, 6.])
xc, yc = x - x.mean(), y - y.mean()
Sxx, Sxy = xc @ xc, xc @ yc
print("== 10A.1  Sxx =", Sxx, " Sxy =", Sxy, " xbar =", x.mean(), " ybar =", y.mean())
for a in (0, 5, 15, 3):
    w = Sxy / (Sxx + a); b = y.mean() - w * x.mean()
    m = Ridge(alpha=a).fit(x[:, None], y) if a > 0 else None
    print(f"alpha = {a:2d}: Sxx + alpha = {Sxx + a:4.0f}  w = {w:.4f}  b = {b:.4f}",
          "" if m is None else f"  sklearn: w = {m.coef_[0]:.4f} b = {m.intercept_:.4f}")
print("alpha with w = 1:", Sxy / 1.0 - Sxx)
r = yc - (Sxy / (Sxx + 5)) * xc; print("residuals at alpha = 5:", r, " sum:", r.sum().round(12), " SSE:", (r @ r).round(4))

print("\n== 10A.2")
G = np.array([[5., 4], [4, 5]]); g = np.array([9., 3])
for a in (0, 1):
    A = G + a * np.eye(2); print(f"alpha = {a}: det = {np.linalg.det(A):.0f}  inverse*det =", (np.linalg.inv(A) * np.linalg.det(A)).tolist(),
                                  " w =", np.linalg.solve(A, g), " exact:", [round(v, 4) for v in np.linalg.solve(A, g)])
lam, V = np.linalg.eigh(G); print("eigenvalues:", lam, " eigenvectors (columns):\n", V)
vg = V.T @ g; print("v_j' g:", vg, " filter alpha=1:", lam / (lam + 1), " df =", (lam / (lam + 1)).sum())
c = vg / (lam + 1); print("coefficients c_j:", c, " w = V c =", V @ c)
print("SVD-form check (c_j = f_j * LS coefficient):", (lam / (lam + 1)) * (vg / lam))

print("\n== 10A.3  orthonormal columns")
rng = np.random.default_rng(0); Z = rng.normal(size=(40, 3)); Z -= Z.mean(0); Q = np.linalg.qr(Z)[0]
w0 = np.array([3., -2, 0.5]); yy = Q @ w0 + 7.0
print("Xc'Xc =\n", Q.T @ Q)
for a in (1, 3):
    m = Ridge(alpha=a).fit(Q, yy); print(f"alpha = {a}: Ridge weights {m.coef_}   w0/(1+alpha) = {w0 / (1 + a)}   df = p/(1+alpha) = {3 / (1 + a)}")

print("\n== 10A.4  leave-one-out, alpha = 5")
a = 5.0; w = Sxy / (Sxx + a); r = yc - w * xc; h = 0.25 + xc ** 2 / (Sxx + a); e = r / (1 - h)
print("w =", w, " xc =", xc, " yc =", yc); print("r =", r, "\nh =", h, "\n1 - h =", 1 - h, "\ne =", e, "\ne^2 =", e ** 2, " mean e^2 =", (e ** 2).mean().round(4))
ex = []
for i in range(4):
    k = np.arange(4) != i; ex.append(y[i] - Ridge(alpha=a).fit(x[k, None], y[k]).predict(x[i:i + 1, None])[0])
print("explicit refits:", np.round(ex, 4))
k = np.arange(4) != 0; xk, yk = x[k], y[k]; xk_c = xk - xk.mean()
wk = (xk_c @ (yk - yk.mean())) / (xk_c @ xk_c + a)
print("refit without sample 1: xbar =", xk.mean(), " ybar =", yk.mean().round(4), " Sxx =", xk_c @ xk_c, " Sxy =", (xk_c @ (yk - yk.mean())).round(4),
      " w =", round(wk, 5), " prediction at 0 =", round(yk.mean() + wk * (0 - xk.mean()), 4), " error =", round(1 - (yk.mean() + wk * (0 - xk.mean())), 4))

print("\n== 10C.1  shifted targets (delivery data), alpha = 10")
xd = np.array([1, 2, 3, 4, 5.]); yd = np.array([3, 4, 6, 7, 10.]); a = 10.0
for shift in (0, 100):
    ys = yd + shift
    m = Ridge(alpha=a).fit(xd[:, None], ys)
    # penalised intercept: ridge on the columns (1, x) without centring
    X1 = np.c_[np.ones(5), xd]; wb = np.linalg.solve(X1.T @ X1 + a * np.eye(2), X1.T @ ys)
    print(f"shift {shift:3d}: unpenalised intercept: w = {m.coef_[0]:.4f} b = {m.intercept_:.4f} |"
          f" penalised intercept: w = {wb[1]:.4f} b = {wb[0]:.4f}  (X'X + aI)=", (X1.T @ X1 + a * np.eye(2)).tolist(), " X'y =", (X1.T @ ys).tolist())
    r = ys - (wb[0] + wb[1] * xd); print("          penalised fit: residual sum =", r.sum().round(3), " mean prediction", (wb[0] + wb[1] * xd).mean().round(3), " mean y", ys.mean())

print("\n== instructor note of 10B.2: noise factor 0.1 instead of 0.01")
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
Xd, td = load_diabetes(return_X_y=True)
for f in (0.01, 0.1):
    rng = np.random.default_rng(0); cp = Xd[:, 2] + f * Xd[:, 2].std() * rng.normal(size=len(Xd))
    Xtr, Xte, ytr, yte = train_test_split(np.c_[Xd, cp], td, test_size=0.25, random_state=0)
    Xc = Xtr - Xtr.mean(0); lam = np.linalg.eigvalsh(Xc.T @ Xc)
    w = LinearRegression().fit(Xtr, ytr).coef_[[2, 10]]
    print(f"noise factor {f}: smallest eigenvalue {lam[0]:.2e}  least-squares weights of bmi and copy {w.round(1)}  sum {w.sum():.1f}")
