# Numbers quoted in the model answers to the questions for discussion of Chapter 12
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.model_selection import train_test_split
np.set_printoptions(precision=4, suppress=True)
x = np.array([-3., -1, 1, 3]); y = np.array([-2., -2, 1, 3]); X = x[:, None]

print("== Q1  decisions: the band at three settings, and the diabetes band")
a, lam = 1.0, 5.0; prec = lam + a * 20
for xs in (0, 2, 6):
    m = a * 18 / prec * xs; sd = np.sqrt(1 / a + xs ** 2 / prec); print(f"x* = {xs}: mean {m:.3f}, sd {sd:.3f}, interval {m - 2 * sd:.2f} to {m + 2 * sd:.2f}")
print("chance that the reading exceeds an upper limit of 6:", end=" ")
from scipy.stats import norm
for xs in (0, 6):
    m = 18 / prec * xs; sd = np.sqrt(1 + xs ** 2 / prec); print(f" x*={xs}: {1 - norm.cdf((6 - m) / sd):.4g};", end="")
print()
d = load_diabetes(); Xtr, Xte, ytr, yte = train_test_split(d.data, d.target, test_size=0.25, random_state=0)
br = BayesianRidge().fit(Xtr, ytr); mu, sd = br.predict(Xte, return_std=True)
noise = 1 / br.alpha_; wpart = sd ** 2 - noise
print("diabetes: predictive sd min %.1f max %.1f ; spread of the means: sd of predictions %.1f, range %.0f to %.0f" % (sd.min(), sd.max(), mu.std(), mu.min(), mu.max()))
print("share of the predictive variance that comes from the weights: mean %.3f, max %.3f" % ((wpart / sd ** 2).mean(), (wpart / sd ** 2).max()))
order_by_mean = np.argsort(mu); order_by_ucb = np.argsort(mu + 2 * sd)
print("Spearman rank correlation between mean and mean + 2 sd:", round(float(np.corrcoef(np.argsort(order_by_mean), np.argsort(order_by_ucb))[0, 1]), 4))

print("\n== Q2  what shrinks with more data and what does not (calibration data copied k times)")
for k in (1, 4, 16, 64, 256):
    b = BayesianRidge().fit(np.tile(X, (k, 1)), np.tile(y, k)); mu6, sd6 = b.predict([[6.0]], return_std=True)
    print(f"k = {k:3d}: alpha_ = {b.alpha_:.4f} floor 1/sqrt(alpha_) = {b.alpha_ ** -0.5:.4f}  sd of slope = {np.sqrt(b.sigma_[0, 0]):.4f}  predictive sd at 6 = {sd6[0]:.4f}  weight part of variance = {sd6[0] ** 2 - 1 / b.alpha_:.4f}")
r_ols = y - 0.9 * x; print("limit: alpha_inf = n / SSE_OLS = 4 /", (r_ols ** 2).sum(), "=", 4 / (r_ols ** 2).sum(), " floor =", ((r_ols ** 2).sum() / 4) ** 0.5)
b16 = BayesianRidge().fit(np.tile(X, (16, 1)), np.tile(y, 16)); fl = b16.alpha_ ** -0.5
print("mean of m repeated readings of the new quantity: floor 1/sqrt(m alpha):", [round(float(fl / np.sqrt(m)), 3) for m in (1, 2, 4, 9)])
print("and noise halved (alpha x 4):", round(fl / 2, 3))

print("\n== Q3  BayesianRidge against Ridge on the diabetes training data")
ratio = br.lambda_ / br.alpha_
rd = Ridge(alpha=ratio).fit(Xtr, ytr)
print("alpha_ = %.3e lambda_ = %.3e ratio lambda_/alpha_ = %.4f  n_iter_ = %d" % (br.alpha_, br.lambda_, ratio, br.n_iter_))
print("max |weights BayesianRidge - Ridge(alpha=ratio)| = %.2e" % np.abs(br.coef_ - rd.coef_).max())
from sklearn.model_selection import cross_val_score
for r in (0.003, ratio, 0.063, 1.0, 10.0):
    mse = -cross_val_score(Ridge(alpha=r), Xtr, ytr, cv=5, scoring="neg_mean_squared_error").mean(); print(f"Ridge(alpha={r:.4f}): five-fold CV mse {mse:.1f}")
print("what Ridge cannot give: noise sd = %.1f, predictive sd %.1f to %.1f" % (br.alpha_ ** -0.5, sd.min(), sd.max()))
print("two pairs with the same ratio lambda/alpha = 5 have the same posterior mean and different spread:")
for s in (1, 10):
    al, la = s * 1.0, s * 5.0; pr = la + al * 20; print(f"  pair ({al}, {la}): mean {al * 18 / pr:.4f}  sd of slope {pr ** -0.5:.4f}")

print("\n== Q4  evidence against training error, calibration data, alpha fixed at 1.667")
al = 5 / 3; n = 4
def logev(la):
    pr = la + al * 20; m = al * 18 / pr; sse = ((y - m * x) ** 2).sum()
    return 0.5 * (np.log(la) + n * np.log(al) - n * np.log(2 * np.pi) - np.log(pr) - al * sse - la * m * m), sse, m, pr
print("   lambda        w      SSE   log evidence   (ln|A| - ln lambda = Occam term)")
for la in (1e-4, 1e-2, 0.1, 0.5, 1.282, 3, 10, 100, 1000):
    le, sse, m, pr = logev(la); print(f"{la:9.4g} {m:8.4f} {sse:8.4f} {le:12.4f}   {np.log(pr) - np.log(la):8.3f}")
las = np.logspace(-4, 3, 2000); les = [logev(l)[0] for l in las]; k = int(np.argmax(les))
print("maximum of the evidence on the grid: lambda = %.3f, log evidence %.4f" % (las[k], les[k]))
print("training error SSE decreases monotonically as lambda falls: ", all(np.diff([logev(l)[1] for l in las]) >= -1e-12))
