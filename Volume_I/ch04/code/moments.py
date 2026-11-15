# Expectation and variance: a die, two dice, a binomial (§4.5).
import numpy as np
from scipy import stats

faces = np.arange(1, 7)
p = np.full(6, 1 / 6)                        # fair die
mean = np.sum(faces * p)                     # Eq. 4.5-1
var = np.sum((faces - mean) ** 2 * p)        # Eq. 4.5-2
print(f"die: E = {mean:.4f}  Var = {var:.4f}  sd = {var**.5:.4f}")
print(f"E[2X+1] = {2*mean + 1:.1f}   Var[2X+1] = {4*var:.4f}")

sums = (faces[:, None] + faces[None, :]).ravel()    # two dice
print(f"two dice: E = {sums.mean():.1f}  Var = {sums.var():.4f}")

bn = stats.binom(5, 0.3)                # 5 Bernoulli(0.3) added
print(f"binomial(5, 0.3): mean {bn.mean():.2f}  var {bn.var():.2f}")
