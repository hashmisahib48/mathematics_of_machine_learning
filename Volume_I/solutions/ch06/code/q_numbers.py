# Numbers quoted in the model answers to the Chapter 6 questions. Output: verify/q_numbers.out
import numpy as np
from numpy.polynomial import Polynomial, chebyshev as Ch
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

rm = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
def draw(rng, n):
    x = np.sort(rng.uniform(0, 1, n)); return x, np.sin(2 * np.pi * x) + rng.normal(0, 0.25, n)

print("== Q1  Listing 6.2 data: training error against 5-fold cross-validation")
rng = np.random.default_rng(7); x_tr, y_tr = draw(rng, 15); x_te, y_te = draw(rng, 400)
folds = np.random.default_rng(0).permutation(15).reshape(5, 3)
print("degree  train   CV-RMSE   test")
for d in (0, 1, 2, 3, 4, 5, 7, 9):
    f = Polynomial.fit(x_tr, y_tr, d); cv = []
    for k in range(5):
        tr = np.setdiff1d(np.arange(15), folds[k]); g = Polynomial.fit(x_tr[tr], y_tr[tr], d)
        cv.append(np.mean((y_tr[folds[k]] - g(x_tr[folds[k]])) ** 2))
    print(f"{d:6d} {rm(y_tr, f(x_tr)):7.3f} {np.sqrt(np.mean(cv)):9.3f} {rm(y_te, f(x_te)):8.3f}")

xv, yv = draw(np.random.default_rng(70), 100)        # 100 fresh validation samples
print("validation RMSE on 100 fresh samples:", {d: round(rm(yv, Polynomial.fit(x_tr, y_tr, d)(xv)), 3) for d in (0, 1, 2, 3, 4, 5, 7, 9)})
print("\n== Q2  best degree from the three terms (exact for fixed x on an even grid, 101 test points)")
x0 = np.linspace(0, 1, 101); f0 = np.sin(2 * np.pi * x0)
def terms(n, deg):
    xt = (np.arange(n) + 0.5) / n; V = Ch.chebvander(2 * xt - 1, deg); V0 = Ch.chebvander(2 * x0 - 1, deg)
    A = np.linalg.pinv(V); bias2 = np.mean((V0 @ (A @ np.sin(2 * np.pi * xt)) - f0) ** 2)
    var1 = np.mean(np.einsum('ij,jk,ik->i', V0, np.linalg.inv(V.T @ V), V0))         # variance for sigma = 1
    return bias2, var1
degs = range(0, 13)
for sigma, n in ((0.25, 20), (0.5, 20), (0.25, 40), (0.25, 80)):
    tot = []
    for d in degs: b2, v1 = terms(n, d); tot.append(sigma**2 + b2 + sigma**2 * v1)
    best = int(np.argmin(tot)); b2, v1 = terms(n, best)
    print(f"sigma {sigma}, n {n:3d}, sigma^2/n = {sigma**2/n:.5f}: best degree {best}, bias^2 {b2:.4f}, variance {sigma**2*v1:.4f}, total {tot[best]:.4f}; totals d=1..6: {np.round(tot[1:7], 4).tolist()}")

for sigma in np.arange(0.25, 4, 0.01):          # how noisy must the data be before degree 3 is too flexible?
    tot = [sigma**2 + terms(20, d)[0] + sigma**2 * terms(20, d)[1] for d in range(0, 9)]
    if int(np.argmin(tot)) < 3: print(f"n = 20: the best degree first drops below 3 at sigma = {sigma:.2f} (best {int(np.argmin(tot))})"); break
print("\n== Q3  three ways to shrink the gap of the degree-9 fit (Listing 6.2 data)")
f9 = Polynomial.fit(x_tr, y_tr, 9); print(f"degree 9, 15 samples:   train {rm(y_tr, f9(x_tr)):.3f} test {rm(y_te, f9(x_te)):.3f}")
f3 = Polynomial.fit(x_tr, y_tr, 3); print(f"degree 3, 15 samples:   train {rm(y_tr, f3(x_tr)):.3f} test {rm(y_te, f3(x_te)):.3f}")
r1 = np.random.default_rng(7); x1, y1 = draw(r1, 100); x2, y2 = draw(r1, 400); g9 = Polynomial.fit(x1, y1, 9)
print(f"degree 9, 100 samples:  train {rm(y1, g9(x1)):.3f} test {rm(y2, g9(x2)):.3f}")
for a in (1e-4, 1e-2, 1e-1, 1.0, 10.0):
    m = make_pipeline(PolynomialFeatures(9, include_bias=False), StandardScaler(), Ridge(alpha=a)).fit(x_tr[:, None], y_tr)
    print(f"degree 9, ridge alpha {a:g}: train {rm(y_tr, m.predict(x_tr[:, None])):.3f} test {rm(y_te, m.predict(x_te[:, None])):.3f}")

print("\n== Q4  the slope of Eq. 6.5-3 for the data of Problem 6A.4")
x = np.array([-2., -1, 1, 2]); y = np.array([-3., -1, 1, 3]); Sxy, Sxx = x @ y, x @ x
for a in (0, 1, 10, 100, 1000, 1e6): print(f"alpha {a:8g}: slope {Sxy/(Sxx+a):.5f}  weight of the data Sxx/(Sxx+alpha) = {Sxx/(Sxx+a):.4f}")
