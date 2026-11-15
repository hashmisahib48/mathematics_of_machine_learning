# Hand checks for Problems 15A.1 - 15A.4 (every number quoted in the solutions)
import numpy as np
from math import comb
from itertools import product
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer
np.set_printoptions(precision=4, suppress=True)

print("== 15A.1  sample (3, -1), degree 2")
x1, x2 = 3.0, -1.0
cols = np.array([1, x1, x2, x1**2, x1*x2, x2**2])
w = np.array([1, 2, -1, 0.5, 1, 2.0])
print("columns:", cols, " sklearn:", PolynomialFeatures(2).fit_transform([[x1, x2]])[0])
print("terms w*phi:", w * cols, " yhat =", (w * cols).sum())
print("exponent triples with sum <= 3 for p = 3, by degree:",
      [sum(1 for a in product(range(4), repeat=3) if sum(a) == k) for k in range(4)],
      " total", sum(1 for a in product(range(4), repeat=3) if sum(a) <= 3),
      " C(6,3) =", comb(6, 3), " sklearn:", PolynomialFeatures(3).fit(np.zeros((1, 3))).n_output_features_)

print("\n== 15A.2  x = 1..5, y = 6, 2, 1, 4, 12")
x = np.arange(1., 6); y = np.array([6., 2, 1, 4, 12]); u = x - 3; q = u**2 - 2
print("u", u, " q", q, " sum u, q, uq:", u.sum(), q.sum(), (u*q).sum())
print("ybar", y.mean(), " u.y", u @ y, " u.u", u @ u, " q.y", q @ y, " q.q", q @ q)
c1 = (u @ y) / (u @ u); c2 = (q @ y) / (q @ q); print("c1, c2 =", c1, c2)
fit = y.mean() + c1*u + c2*q; r = y - fit
print("uy", u*y, " qy", q*y); print("fitted", fit, " residuals", r, " SSE", (r**2).sum())
# expansion of ybar + c1 (x-3) + c2 ((x-3)^2 - 2)
b2 = c2; b1 = c1 - 6*c2; b0 = y.mean() - 3*c1 + 9*c2 - 2*c2
print("weights of 1, x, x^2:", b0, b1, b2, " lstsq:", np.linalg.lstsq(np.c_[np.ones(5), x, x**2], y, rcond=None)[0])
print("pieces: 5 - 3*1.4 + 9*2 - 2*2 =", 5 - 3*c1 + 9*c2 - 2*c2, "  1.4 - 6*2 =", c1 - 6*c2)
print("prediction at 6:", b0 + b1*6 + b2*36, " (u=3, q=7):", y.mean() + c1*3 + c2*7)
sst = ((y - y.mean())**2).sum(); print("SST", sst, " R2", 1 - (r**2).sum()/sst)
print("orthogonality of residuals: sum r, sum x r, sum x^2 r =", r.sum(), x @ r, x**2 @ r)
line = np.polyfit(x, y, 1); print("straight line:", line, " SSE", ((y - np.polyval(line, x))**2).sum())

print("\n== 15A.3  cubic B-spline, knots 0..4")
def B(t, j, k, xv):             # Cox-de Boor, Eq. 15.4-1 and 15.4-2; t: knots, j: first knot, k: degree
    if k == 0: return 1.0 if t[j] <= xv < t[j+1] else 0.0
    a = (xv - t[j]) / (t[j+k] - t[j]); c = (t[j+k+1] - xv) / (t[j+k+1] - t[j+1])
    return a * B(t, j, k-1, xv) + c * B(t, j+1, k-1, xv)
t = np.arange(-3., 8.)          # knots of the transformer: -3 ... 7
print("hat on (1,2,3) at 2:", B(t, 4, 1, 2.0), " hat on (0,1,2) at 2:", B(t, 3, 1, 2.0))
q1 = B(t, 3, 2, 2.0); q2 = B(t, 4, 2, 2.0)
print("quadratic on (0,1,2,3) at 2:", q1, " quadratic on (1,2,3,4) at 2:", q2)
print("first fraction (x-t0)/(t3-t0) =", (2-0)/(3-0), " second (t4-x)/(t4-t1) =", (4-2)/(4-1))
print("B(2) =", (2/3)*q1 + (2/3)*q2, " direct recursion:", B(t, 3, 3, 2.0))
print("cubic on (0..4) at 1 and 3:", B(t, 3, 3, 1.0), B(t, 3, 3, 3.0), " 1/6 =", 1/6)
print("quadratics at 1: on (0,1,2,3)", B(t, 3, 2, 1.0), " on (1,2,3,4)", B(t, 4, 2, 1.0))
print("row of the seven cubic columns at 2 (recursion):", np.array([B(t, j, 3, 2.0) for j in range(7)]))
sp = SplineTransformer(knots=np.arange(5.)[:, None], degree=3).fit(np.zeros((2, 1)))
print("SplineTransformer row at 2:", sp.transform([[2.0]])[0], " sum", sp.transform([[2.0]])[0].sum())
print("knots used by the transformer:", sp.bsplines_[0].t)

print("\n== 15A.4  counting")
print("(a) C(10,4) =", comb(10, 4), " sklearn", PolynomialFeatures(4).fit(np.zeros((1, 6))).n_output_features_)
print("(b) 1 + 6 + 15 + 20 =", sum(comb(6, k) for k in range(4)), [comb(6, k) for k in range(4)],
      " sklearn", PolynomialFeatures(3, interaction_only=True).fit(np.zeros((1, 6))).n_output_features_)
print("(c) 4 * (7 + 3 - 1 - 1) =", 4*(7+3-1-1), " per feature with bias:", 7+3-1,
      " sklearn", SplineTransformer(n_knots=7, degree=3, include_bias=False).fit(np.zeros((8, 4))).n_features_out_)
