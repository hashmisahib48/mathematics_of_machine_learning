# One log-loss round on the ten loan applicants (income only).
import numpy as np
from scipy.special import expit

eta = 0.5
np.set_printoptions(linewidth=60)
inc = np.array([22, 40, 55, 28, 62, 18, 36, 26, 48, 75.])
z = np.array([0, 1, 0, 0, 1, 0, 1, 0, 1, 1.])      # 1 = repaid
F0 = np.log(z.mean() / (1 - z.mean()))             # Eq. 27.5-4
p = expit(np.full(10, F0))                         # all 0.5
r = z - p                                          # Eq. 27.5-3
left = inc <= 32                                   # the split
gam = [r[k].sum() / (p[k] * (1 - p[k])).sum()      # Eq. 27.5-6
       for k in (left, ~left)]
print("F0 =", F0, " r =", r)
print("sum r (left, right):", r[left].sum(), r[~left].sum(),
      " sum p(1-p):", (p * (1 - p))[left].sum(),
      (p * (1 - p))[~left].sum())
print("leaf values", np.round(gam, 4))
F1 = F0 + eta * np.where(left, gam[0], gam[1])
print("F1 =", F1.round(4))
print("p  =", expit(F1).round(4))
q = expit(F1)
loss = -np.mean(z * np.log(q) + (1 - z) * np.log(1 - q))
print("log loss before", np.log(2).round(4), "after", loss.round(4))
