# Numbers quoted in the model answers to the questions of Chapter 14
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE + '/..')
import numpy as np
from scipy.special import digamma
from sklearn.linear_model import PoissonRegressor, GammaRegressor, LinearRegression, TweedieRegressor
from sklearn.metrics import mean_tweedie_deviance
np.set_printoptions(precision=4, suppress=True)
dG = lambda y, m: 2 * ((y - m) / m - np.log(y / m))

print("== Q1  log y regression against the gamma GLM")
k = 1.2
print("exp(E ln y)/E y for a gamma variable of shape %.1f: exp(psi(k))/k = %.4f" % (k, np.exp(digamma(k)) / k))
for r in (10, 0.1):
    print("ratio y/mu = %5.1f: gamma unit deviance %.3f, squared log error %.3f" % (r, dG(r, 1.0), np.log(r) ** 2))
rat_g, rat_l, tot_g, tot_l = [], [], [], []
for seed in range(200):
    rng = np.random.default_rng(1000 + seed); xs = rng.uniform(0, 4, 70); ys = rng.gamma(1.2, np.exp(0.6 + 0.55 * xs) / 1.2)
    g = GammaRegressor(alpha=0, tol=1e-10, max_iter=500).fit(xs[:, None], ys); c = np.polyfit(xs, np.log(ys), 1)
    t = np.exp(0.6 + 0.55 * 2.0); rat_g.append(g.predict([[2.0]])[0] / t); rat_l.append(np.exp(c[1] + 2.0 * c[0]) / t)
    tot_g.append(g.predict(xs[:, None]).sum() / ys.sum()); tot_l.append(np.exp(c[1] + c[0] * xs).sum() / ys.sum())
print("200 data sets of 70 points (shape 1.2): prediction / true mean at x = 2: gamma GLM mean %.3f, exp of log-OLS mean %.3f" % (np.mean(rat_g), np.mean(rat_l)))
print("predicted total / observed total: gamma GLM %.3f, exp of log-OLS %.3f" % (np.mean(tot_g), np.mean(tot_l)))
for ln in open('../../ch14/verify/case_tweedie.out'): 
    if 'zero' in ln: print("the chapter's own output:", ln.strip())

print("\n== Q2  alpha = 1 and the unit of the feature")
x0 = np.arange(5.)[:, None]; y = np.array([1., 2, 2, 4, 6])
for unit, f in (("10^4 cycles (x 10)", 10.0), ("10^5 cycles (x 1)", 1.0), ("10^6 cycles (x 0.1)", 0.1)):
    m = PoissonRegressor(alpha=1.0, tol=1e-10, max_iter=1000).fit(f * x0, y)
    print("unit %-20s alpha = 1: coef_ = %.4f -> per block of 10^5 cycles %.4f ; D2 = %.4f" % (unit, m.coef_[0], m.coef_[0] * f, m.score(f * x0, y)))
m0 = PoissonRegressor(alpha=0.0, tol=1e-10).fit(x0, y); print("alpha = 0: per block %.4f, D2 %.4f" % (m0.coef_[0], m0.score(x0, y)))
xs_ = (x0 - x0.mean()) / x0.std(); ms = PoissonRegressor(alpha=1.0, tol=1e-10).fit(xs_, y); print("standardised feature, alpha = 1: per block %.4f (coef %.4f per sd), D2 %.4f" % (ms.coef_[0] / x0.std(), ms.coef_[0], ms.score(xs_, y)))

print("\n== Q3  exposure")
rng = np.random.default_rng(3); n = 3000; area = rng.uniform(0.1, 10, n); xf = rng.normal(size=n); lam = np.exp(0.2 + 0.5 * xf)
N = rng.poisson(area * lam); X = xf[:, None]
a = PoissonRegressor(alpha=0, tol=1e-12, max_iter=1000).fit(X, N / area, sample_weight=area)
b = PoissonRegressor(alpha=0, tol=1e-12, max_iter=1000).fit(X, N)
X2 = np.c_[xf, np.log(area)]; c = PoissonRegressor(alpha=0, tol=1e-12, max_iter=1000).fit(X2, N)
print("true: intercept 0.2, weight 0.5")
print("rate target + weight = area : b %.4f w %.4f" % (a.intercept_, a.coef_[0]))
print("raw counts, no exposure     : b %.4f w %.4f" % (b.intercept_, b.coef_[0]))
print("raw counts + ln(area) as a feature: b %.4f w %.4f, weight of ln(area) %.4f" % (c.intercept_, c.coef_[0], c.coef_[1]))
# the same fit by Newton steps on the count likelihood with ln(area) as an offset
Z = np.c_[np.ones(n), xf]; beta = np.zeros(2); off = np.log(area)
for _ in range(50):
    mu = np.exp(Z @ beta + off); g = Z.T @ (N - mu); H = Z.T @ (mu[:, None] * Z); beta = beta + np.linalg.solve(H, g)
print("offset model (Newton):      b %.6f w %.6f ; difference from the weighted fit %.1e" % (beta[0], beta[1], max(abs(beta[0] - a.intercept_), abs(beta[1] - a.coef_[0]))))
print("area range %.2f to %.2f; the largest and smallest area carry the weights %.2f and %.2f" % (area.min(), area.max(), area.max(), area.min()))

print("\n== Q4  overdispersion: Var = 3 mean")
def sim(seed, n=2000, phi=3.0):
    r = np.random.default_rng(seed); x = r.uniform(0, 3, n); mu = np.exp(0.3 + 0.6 * x)
    shape = mu / (phi - 1); yv = r.poisson(r.gamma(shape, (phi - 1) * np.ones(n)))       # negative binomial, Var = phi mu
    return x[:, None], yv, mu
ws, ses, disp = [], [], []
for s in range(200):
    X, yv, mu = sim(s); m = PoissonRegressor(alpha=0, tol=1e-10, max_iter=1000).fit(X, yv); mh = m.predict(X)
    Z = np.c_[np.ones(len(yv)), X[:, 0]]; H = Z.T @ (mh[:, None] * Z); se = np.sqrt(np.diag(np.linalg.inv(H)))
    ws.append([m.intercept_, m.coef_[0]]); ses.append(se); disp.append(np.sum((yv - mh) ** 2 / mh) / (len(yv) - 2))
ws = np.array(ws); ses = np.array(ses)
print("true weight 0.6; mean estimate %.4f; sd of the estimate over 200 data sets %.4f; mean Poisson standard error %.4f; ratio %.2f (sqrt 3 = %.2f)" % (ws[:, 1].mean(), ws[:, 1].std(), ses[:, 1].mean(), ws[:, 1].std() / ses[:, 1].mean(), np.sqrt(3)))
print("Pearson chi2/(n - p): mean %.2f, range %.2f to %.2f" % (np.mean(disp), min(disp), max(disp)))
cover = np.mean(abs(ws[:, 1] - 0.6) < 1.96 * ses[:, 1]); print("share of data sets in which the nominal 95%% interval of the weight holds the true value: %.2f" % cover)
X, yv, mu = sim(0); m = PoissonRegressor(alpha=0, tol=1e-10, max_iter=1000).fit(X, yv); print("one data set: fitted total %.0f, observed total %d" % (m.predict(X).sum(), yv.sum()))
print("\n== Q5  deviances of different powers")
yv0 = np.array([4.]); mv0 = np.array([2.])
for p in (0, 1, 1.5, 2, 3):
    d1 = mean_tweedie_deviance(yv0, mv0, power=p); d10 = mean_tweedie_deviance(10 * yv0, 10 * mv0, power=p)
    print("power %.1f: d(4, 2) = %8.4f   d(40, 20) = %9.4f   ratio %.4g  (10^(2-p) = %.4g)" % (p, d1, d10, d10 / d1, 10 ** (2 - p)))
rng = np.random.default_rng(1); x = rng.uniform(0, 1, 500); yy = rng.gamma(2, np.exp(1 + x) / 2)
for p in (1.0, 1.5, 2.0):
    m = TweedieRegressor(power=p, alpha=0, tol=1e-10, max_iter=1000).fit(x[:, None], yy); mu = m.predict(x[:, None])
    print("model of power %.1f: deviance in its own power %.4f, in power 1: %.4f, in power 1.5: %.4f, in power 2: %.4f" % (p, mean_tweedie_deviance(yy, mu, power=p), *(mean_tweedie_deviance(yy, mu, power=q) for q in (1.0, 1.5, 2.0))))
