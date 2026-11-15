# Numbers quoted in the model answers to Questions 1 - 4 of Chapter 15
import numpy as np
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler
np.set_printoptions(precision=4, suppress=True)
x = np.arange(1., 6); y = np.array([2., 3, 8, 15, 22])          # battery data of the chapter

print("== Q1  linear in the weights")
Phi = np.c_[np.ones(5), x, x**2]
w = np.linalg.solve(Phi.T @ Phi, Phi.T @ y); w2 = np.linalg.solve(Phi.T @ Phi, Phi.T @ (2*y))
print("weights (b, w1, w2):", w, " for 2y:", w2, " ratio:", w2 / w)
print("cond of Phi'Phi:", np.linalg.cond(Phi.T @ Phi))
f = lambda x, a, c: a * np.exp(c * x)
for yy, nm in ((y, "y"), (2*y, "2y")):
    p, _ = curve_fit(f, x, yy, p0=[1, 0.5], full_output=False)
    _, _, info, *_ = curve_fit(f, x, yy, p0=[1, 0.5], full_output=True)
    print(f"exponential a*exp(c x) fitted to {nm}: a, c =", p, " function evaluations:", info['nfev'])
print("sum of squares, quadratic model as a function of w2 (others at optimum): min at", w[2],
      " SSE there", ((y - Phi @ w)**2).sum(), " at w2 +/- 0.5:", [((y - Phi @ (w + np.array([0, 0, d])))**2).sum() for d in (-.5, .5)])

print("\n== Q2  StandardScaler after PolynomialFeatures (30 yearly values 1990-2019, as Listing 15.5)")
year = np.arange(1990, 2020.0)[:, None]
rng = np.random.default_rng(0); t = year[:, 0] - 1990; yy = 5 + 0.3*t + 0.02*t**2 + rng.normal(0, 0.5, 30)
P = PolynomialFeatures(3).fit_transform(year)                      # with the column of ones
sc = StandardScaler().fit(P); Z = sc.transform(P)
print("column of ones: mean", sc.mean_[0], " scale_", sc.scale_[0], " after scaling:", np.unique(Z[:, 0]))
for d in (2, 3, 4):
    A = PolynomialFeatures(d, include_bias=False).fit_transform(year)
    after = StandardScaler().fit_transform(A)                       # wrong order
    before = PolynomialFeatures(d, include_bias=False).fit_transform(StandardScaler().fit_transform(year))
    cor = np.corrcoef(after.T)[0, 1]
    print(f"degree {d}: cond scaled-after {np.linalg.cond(after):.3e}  scaled-before {np.linalg.cond(before):.3e}"
          f"  corr(x, x^2) after scaling = {cor:.6f}")
    m = make_pipeline(PolynomialFeatures(d, include_bias=False), StandardScaler(), LinearRegression()).fit(year, yy)
    m2 = make_pipeline(StandardScaler(), PolynomialFeatures(d, include_bias=False), LinearRegression()).fit(year, yy)
    print(f"          rank_ {m[-1].rank_} (scaler after) against {m2[-1].rank_} (scaler before)")
print("number of columns for p = 50, d = 3:", PolynomialFeatures(3).fit(np.zeros((1, 50))).n_output_features_, "- the scaler does not change it")

print("\n== Q3  extrapolation of a cubic spline beyond speed 5")
for rule in ("constant", "linear", "continue"):
    m = make_pipeline(SplineTransformer(n_knots=3, degree=3, extrapolation=rule, include_bias=False),
                      LinearRegression()).fit(x[:, None], y)
    print(f"{rule:9s} at 5, 6, 7, 8:", m.predict([[5.], [6.], [7.], [8.]]), " R2 on data", round(m.score(x[:, None], y), 4))
line = np.polyfit(x, y, 1); par = np.polyfit(x, y, 2)
print("straight line at 6, 8:", np.polyval(line, [6, 8]), " parabola at 6, 8:", np.polyval(par, [6, 8]))
print("slope of the fitted parabola at 5:", 2*par[0]*5 + par[1], " of the line:", line[0])

print("\n== Q4  banded normal equations and a distant outlier")
rng = np.random.default_rng(0); n, m_ = 500, 40
xs = np.sort(rng.uniform(0, 10, n))[:, None]; ys = np.sin(xs[:, 0]) + 0.3*rng.normal(size=n)
spl = SplineTransformer(n_knots=m_ - 2, degree=3, include_bias=False).fit(xs)
Phi = spl.transform(xs); G = Phi.T @ Phi
band = max(abs(i - j) for i in range(G.shape[0]) for j in range(G.shape[1]) if abs(G[i, j]) > 1e-12)
print("columns", Phi.shape[1], " max non-zeros per row of Phi:", (Phi > 0).sum(1).max(),
      " non-zero share of Phi'Phi: %.3f" % ((np.abs(G) > 1e-12).mean()), " bandwidth:", band)
print("flops, dense solve ~ m^3 =", Phi.shape[1]**3, " banded ~ m*b^2 =", Phi.shape[1] * band**2)
def fit_pair(X, Y):
    sp_ = make_pipeline(SplineTransformer(n_knots=m_ - 2, degree=3, include_bias=False), Ridge(alpha=1e-6)).fit(X, Y)
    po_ = make_pipeline(StandardScaler(), PolynomialFeatures(12, include_bias=False), Ridge(alpha=1e-6)).fit(X, Y)
    return sp_, po_
s0, p0 = fit_pair(xs, ys)
xo = np.vstack([xs, [[9.9]]]); yo = np.r_[ys, ys[-1] + 20.0]       # one outlier at x = 9.9, 20 above the curve
s1, p1 = fit_pair(xo, yo)
g = np.linspace(0, 5, 200)[:, None]
print("largest change of the fit on [0, 5] after adding the outlier at 9.9 (+20):  spline %.2e   degree-12 polynomial %.3f"
      % (np.abs(s1.predict(g) - s0.predict(g)).max(), np.abs(p1.predict(g) - p0.predict(g)).max()))
g2 = np.linspace(9, 10, 50)[:, None]
print("largest change on [9, 10]:  spline %.2f   polynomial %.2f" % (np.abs(s1.predict(g2) - s0.predict(g2)).max(), np.abs(p1.predict(g2) - p0.predict(g2)).max()))
