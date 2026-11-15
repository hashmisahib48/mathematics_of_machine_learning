# Problem 10C.3: more features than samples (the data of Figure 10C.3)
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
np.set_printoptions(precision=4, suppress=True)
Xw, yw = make_regression(n_samples=60, n_features=100, n_informative=100, noise=20, random_state=0)
Xa, ya, Xb, yb = Xw[:30], yw[:30], Xw[30:], yw[30:]
Xc, yc = Xa - Xa.mean(0), ya - ya.mean()
U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
print("n = 30, p = 100.  rank of the centred X:", int((s > 1e-10).sum()), " eigenvalues of Xc'Xc, number zero:", 100 - int((s > 1e-10).sum()))
lam = np.linalg.eigvalsh(Xc.T @ Xc); print("smallest eigenvalues of Xc'Xc:", lam[:2], "  of Xc'Xc + 1e-3 I:", lam[:2] + 1e-3)
print("\n alpha      train R2   test R2    df     |w|")
for a in (1e-3, 1e-1, 1, 10, 31.62, 100, 1000, 1e5):
    m = Ridge(alpha=a).fit(Xa, ya); f = s ** 2 / (s ** 2 + a)
    print(f"{a:9.3f} {m.score(Xa, ya):9.3f} {m.score(Xb, yb):9.3f} {f.sum():8.2f} {np.linalg.norm(m.coef_):8.2f}")
al = np.logspace(-3, 3, 25); te = [Ridge(alpha=a).fit(Xa, ya).score(Xb, yb) for a in al]
print("best test R2 on the grid of Figure 10C.3: %.3f at alpha = %.2f" % (max(te), al[int(np.argmax(te))]))
print("test R2 at the smallest alpha of the grid: %.3f" % te[0])
# limits
w_pinv = np.linalg.pinv(Xc, rcond=1e-10) @ yc
m = Ridge(alpha=1e-8, solver='svd').fit(Xa, ya)
print("min-norm solution |w| = %.2f" % np.linalg.norm(w_pinv))
print("\nalpha -> 0: largest |Ridge(1e-8, svd) - pinv solution| =", np.abs(m.coef_ - w_pinv).max().round(8), "  fit is exact:", np.allclose(Xc @ w_pinv, yc))
a = 1e8; m = Ridge(alpha=a).fit(Xa, ya)
print("alpha -> infinity: |w - Xc'yc/alpha| / |w| =", (np.linalg.norm(m.coef_ - Xc.T @ yc / a) / np.linalg.norm(m.coef_)).round(6),
      "  predictions minus ybar: max |.| =", np.abs(m.predict(Xb) - ya.mean()).max().round(5))
# dual form, Eq. 10.3-8
a = 10.0; w_dual = Xc.T @ np.linalg.solve(Xc @ Xc.T + a * np.eye(30), yc); print("dual (30 x 30) form agrees with Ridge(10):", np.allclose(w_dual, Ridge(alpha=a).fit(Xa, ya).coef_))
