# Problem 29B.1: majority of five correlated voters, by simulation
import os, sys
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here + "/../../../ch29/code")
import numpy as np
from condorcet import majority      # Eq. 29.3-3, beta-binomial

p, M, n = 0.7, 5, 400000
rng = np.random.default_rng(0)
print("rho  simulated  formula  P(q<.5)  sd(q)  right|q<.5"
      "  right|q>.5")
print(f"0.0  {majority(M, p):9.4f} {majority(M, p):8.4f}")
for rho in (0.1, 0.3):
    s = (1 - rho) / rho                       # a + b
    q = rng.beta(p * s, (1 - p) * s, n)       # case difficulty
    right = rng.random((n, M)) < q[:, None]   # five votes per case
    maj = right.sum(axis=1) > M / 2           # majority is right
    low = q < 0.5
    print(f"{rho:.1f}  {maj.mean():9.4f} {majority(M, p, rho):8.4f}"
          f" {low.mean():8.4f} {q.std():6.3f} {maj[low].mean():10.4f}"
          f" {maj[~low].mean():11.4f}")
    se = np.sqrt(maj.mean() * (1 - maj.mean()) / n)
    print(f"      standard error of the simulated share {se:.4f}")
    r2 = np.corrcoef(right[:, 0], right[:, 1])[0, 1]
    print(f"      correlation of two voters' correctness {r2:.3f}")
