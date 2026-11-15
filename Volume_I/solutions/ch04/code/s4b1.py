# Problem 4B.1: calls per minute, Poisson with mean 3
import numpy as np
from scipy import stats

lam = 3
P2 = stats.poisson.pmf(2, lam)        # exactly two, Eq. 4.3-2
P1 = stats.poisson.sf(0, lam)         # more than 0: at least 1
print(f"exact : P(N = 2) = {P2:.4f}   P(N >= 1) = {P1:.4f}")
print(f"check : 3^2 e^-3/2! = {9 * np.exp(-3) / 2:.4f}",
      f"  1 - e^-3 = {1 - np.exp(-3):.4f}")

rng = np.random.default_rng(0)
sm = rng.poisson(lam, 100000)         # 100 000 simulated minutes
f2, f1 = np.mean(sm == 2), np.mean(sm >= 1)
print(f"simul.: fraction 2 = {f2:.4f}  fraction >= 1 = {f1:.4f}")
for name, p, f in (("N = 2", P2, f2), ("N >= 1", P1, f1)):
    se = np.sqrt(p * (1 - p) / sm.size)   # std. error
    print(f"{name:7s} error {f - p:+.5f} = {(f - p) / se:+.2f} se")
print(f"sample mean {sm.mean():.3f}  variance {sm.var():.3f}")
