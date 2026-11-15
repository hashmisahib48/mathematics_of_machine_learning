# Named distributions with scipy.stats (§4.3 and §4.4).
from scipy import stats

b = stats.binom(5, 0.3)                      # n = 5, theta = 0.3
print(f"P(X=2) binomial   = {b.pmf(2):.4f}")     # Eq. 4.3-1
print(f"P(X<=2) binomial  = {b.cdf(2):.4f}")
po = stats.poisson(2)                        # rate 2 (scipy: mu)
print(f"P(N=0) Poisson    = {po.pmf(0):.4f}")     # Eq. 4.3-2
print(f"P(N>=3) Poisson   = {po.sf(2):.4f}")

z = stats.norm()                             # 68-95-99.7 rule
for k in (1, 2, 3):
    print(f"P(|Z|<{k}) = {z.cdf(k) - z.cdf(-k):.4f}")
h = stats.norm(170, 8)                       # heights, N(170, 8^2)
p = h.cdf(186) - h.cdf(162)
print(f"P(162<H<186)      = {p:.4f}")
e = stats.expon(scale=1 / 0.5)               # scale = 1/rate
print(f"P(T>3) exponential= {e.sf(3):.4f}   mean {e.mean():.1f}")
g = stats.gamma(a=3, scale=1 / 2)            # shape 3, rate 2
print(f"gamma mean, var   = {g.mean():.2f}, {g.var():.2f}")
