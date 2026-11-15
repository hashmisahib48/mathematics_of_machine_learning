# Numbers quoted in the model answers to the questions of Chapter 9
import numpy as np
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
np.set_printoptions(precision=4, suppress=True)
x = np.array([1, 2, 3, 4, 5.]); y = np.array([3, 4, 6, 7, 10.])

print("== Q1  vertical (OLS) against perpendicular (total least squares)")
def tls(x, y):                       # line through the means along the first principal axis
    Z = np.c_[x - x.mean(), y - y.mean()]
    v = np.linalg.svd(Z, full_matrices=False)[2][0]
    w = v[1] / v[0]; return y.mean() - w * x.mean(), w
for unit, f in (("km", 1.0), ("m", 1000.0)):
    xs = f * x
    b0, w0 = np.polyfit(xs, y, 1)[::-1]; b1, w1 = tls(xs, y)
    print(f"x in {unit:2s}: OLS w = {w0:.6f} b = {b0:.4f} pred(6 km) = {b0 + w0 * 6 * f:.3f} |"
          f" perpendicular w = {w1:.6f} b = {b1:.4f} pred(6 km) = {b1 + w1 * 6 * f:.3f}")
Sxx, Syy, Sxy = 10., 30., 17.
print("closed form slope:", (Syy - Sxx + np.hypot(Syy - Sxx, 2 * Sxy)) / (2 * Sxy))

print("\n== Q2  absolute values instead of squares")
def lad(yv):
    m = QuantileRegressor(quantile=0.5, alpha=0.0, solver="highs").fit(x[:, None], yv)
    return m.coef_[0], m.intercept_
for v in (10, 20):
    yv = y.copy(); yv[4] = v
    ols = LinearRegression().fit(x[:, None], yv)
    wl, bl = lad(yv)
    print(f"y5 = {v}: OLS w = {ols.coef_[0]:.3f} b = {ols.intercept_:.3f} pred(6) = {ols.predict([[6]])[0]:.2f} |"
          f" LAD w = {wl:.3f} b = {bl:.3f} pred(6) = {bl + 6 * wl:.2f}  SAD = {np.abs(yv - bl - wl * x).sum():.3f}")
# the absolute-value loss along w (b chosen as the median of the residuals) is piecewise linear
wg = np.round(np.arange(1.0, 2.61, 0.05), 2)
sad = np.array([np.abs(y - w * x - np.median(y - w * x)).sum() for w in wg])
print("clean data: smallest absolute-value loss", sad.min().round(4), "reached for w in", wg[sad < sad.min() + 1e-9].tolist())
print("loss at w = 1.0, 1.5, 2.0, 2.5:", [round(float(sad[list(wg).index(w)]), 3) for w in (1.0, 1.5, 2.0, 2.5)])

print("\n== Q3  km against metres")
for f in (1, 1000):
    m = LinearRegression().fit(f * x[:, None], y)
    r = y - m.predict(f * x[:, None])
    print(f"x*{f:4d}: w = {m.coef_[0]:.4f} b = {m.intercept_:.4f} SSE = {(r ** 2).sum():.4f} R2 = {m.score(f * x[:, None], y):.4f}"
          f" pred(6 km) = {m.predict([[6 * f]])[0]:.3f}  Sxx = {((f * x - f * 3) ** 2).sum():.0f}")

print("\n== Q4  no hyperparameter: complexity enters through the features")
d = load_diabetes(); X, t = d.data, d.target
Xtr, Xte, ytr, yte = train_test_split(X, t, test_size=0.25, random_state=0)
for deg in (1, 2):
    P = PolynomialFeatures(deg, include_bias=False).fit(Xtr)
    m = LinearRegression().fit(P.transform(Xtr), ytr)
    print(f"degree {deg}: {P.n_output_features_:3d} features  train R2 = {m.score(P.transform(Xtr), ytr):.3f}  test R2 = {m.score(P.transform(Xte), yte):.3f}")

print("\n== Q5  ten times as many patients (synthetic stand-in built from the diabetes fit)")
full = LinearRegression().fit(X, t); sig = np.sqrt(((t - full.predict(X)) ** 2).sum() / (len(t) - 11))
C = np.cov(X.T); L = np.linalg.cholesky(C)
print(f"population R2 of the stand-in: {1 - sig ** 2 / (full.coef_ @ C @ full.coef_ + sig ** 2):.3f}   sigma = {sig:.1f}")
rng = np.random.default_rng(0)
def sim(n, reps=200, nte=5000):
    tr, te = [], []
    for _ in range(reps):
        A = rng.normal(size=(n, 10)) @ L.T; yA = full.intercept_ + A @ full.coef_ + rng.normal(0, sig, n)
        B = rng.normal(size=(nte, 10)) @ L.T; yB = full.intercept_ + B @ full.coef_ + rng.normal(0, sig, nte)
        m = LinearRegression().fit(A, yA); tr.append(m.score(A, yA)); te.append(m.score(B, yB))
    return np.mean(tr), np.mean(te)
for n in (331, 3310):
    a, b = sim(n); print(f"n = {n:5d}: mean train R2 = {a:.3f}  mean test R2 = {b:.3f}")
