# Numbers for the model answers to the questions for discussion, Chapter 4
import numpy as np
from scipy import stats

print("== Q1  accuracy and the base rate")
prev, sens, spec = 0.01, 0.95, 0.95
print("accuracy of a test with sens = spec = 0.95 at prevalence 1 %:", sens * prev + spec * (1 - prev))
ppv = sens * prev / (sens * prev + (1 - spec) * (1 - prev)); print("P(sick | +) =", ppv)
print("accuracy of 'always negative' at prevalence 1 %:", 1 - prev, "  sensitivity 0")
print("Example 4.2-1: P(sick | +) =", 0.9 * 0.01 / (0.9 * 0.01 + 0.05 * 0.99))

print("\n== Q2  binomial or Poisson")
n, th = 5, 0.3; print("binomial(5, 0.3): mean", n * th, " variance", n * th * (1 - th), " ratio", (1 - th))
print("Poisson(3): mean 3 variance 3 ratio 1")
for n, th in ((10, 0.3), (100, 0.03), (1000, 0.003)):
    print(f"binomial({n}, {th}) P(2) = {stats.binom.pmf(2, n, th):.4f}   Poisson(3) P(2) = {stats.poisson.pmf(2, 3):.4f}   variance/mean = {1 - th}")

print("\n== Q3  zero correlation, strong dependence")
rng = np.random.default_rng(3); x = rng.uniform(-2, 2, 120); y = x ** 2 + rng.normal(0, 0.2, 120)
print("r(x, y) =", np.corrcoef(x, y)[0, 1], " r(x^2, y) =", np.corrcoef(x ** 2, y)[0, 1], " r(|x|, y) =", np.corrcoef(np.abs(x), y)[0, 1])
print("E[x^3] for symmetric x is 0; mean of x^3 in the sample:", np.mean(x ** 3), " mean x:", x.mean())
xn = rng.normal(size=100000); yn = 0.0 * xn + rng.normal(size=100000)
print("jointly normal and uncorrelated: r =", np.corrcoef(xn, yn)[0, 1], " r(x^2, y^2) =", np.corrcoef(xn ** 2, yn ** 2)[0, 1])

print("\n== Q4  the divisor n")
for n in (5, 10, 100, 1000):
    bias = -1 / n; sd = np.sqrt(2 / (n - 1))
    mse_n1 = 2 / (n - 1); mse_n = (2 * (n - 1) + 1) / n ** 2
    print(f"n = {n:4d}: E[s2_n] = {(n - 1) / n:.4f}  bias = {bias:+.4f}  sd of s2_(n-1) = {sd:.3f}  MSE divisor n-1 = {mse_n1:.4f}  divisor n = {mse_n:.4f}")
r = np.random.default_rng(0); S = r.normal(size=(200000, 5)); v1 = S.var(axis=1, ddof=1); v0 = S.var(axis=1)
print("simulation n = 5: mean ddof=1:", v1.mean(), " ddof=0:", v0.mean(), " MSE ddof=1:", ((v1 - 1) ** 2).mean(), " ddof=0:", ((v0 - 1) ** 2).mean())

print("\n== Q5  the prior variance grows")
xbar, n, s2 = 3.0, 4, 4.0
for t2 in (1.0, 10.0, 100.0, 1e4, 1e6):
    pm = (n / s2 * xbar + 0 / t2) / (n / s2 + 1 / t2); pv = 1 / (n / s2 + 1 / t2); a = s2 / t2
    print(f"tau^2 = {t2:9.0f}: alpha = {a:.6f}  posterior mean = {pm:.4f}  variance = {pv:.4f}")
print("sigma^2/n =", s2 / n)

print("\n== Q6  logarithms")
p = np.full(1000, 0.1); print("product of 1000 probabilities 0.1:", np.prod(p), " sum of logs:", np.log(p).sum(), " exact log10 of the product: -1000")
print("smallest positive double:", np.finfo(float).tiny, " 0.1^308 =", 0.1 ** 308, " 0.1^324 =", 0.1 ** 324)
d = stats.norm.pdf(np.random.default_rng(0).normal(size=1000)); print("1000 normal density values: min", d.min(), " product", np.prod(d), " sum of logs", np.log(d).sum())
