# Majority of m voters of accuracy p: independent, and correlated.
import json
import numpy as np
from scipy.stats import beta, betabinom, binom

def majority(m, p, rho=0.0):
    """P(more than half of m (odd) voters are right), Eq. 29.3-3."""
    if rho == 0:
        return binom.sf(m // 2, m, p)
    s = (1 - rho) / rho                     # a + b of Eq. 29.10-1
    return betabinom.sf(m // 2, m, p * s, (1 - p) * s)

if __name__ == "__main__":
    ms = (1, 3, 5, 11, 21, 101)
    print("independent voters: P(majority right)")
    print("  p     " + "".join(f"{m:8d}" for m in ms))
    for p in (0.55, 0.6, 0.7, 0.45):
        print(f"  {p:.2f}  " + "".join(f"{majority(m, p):8.4f}" for m in ms))
    print("p = 0.7, correlation rho between voters")
    for rho in (0.1, 0.3, 0.6):
        lim = beta.sf(0.5, 0.7 * (1 - rho) / rho, 0.3 * (1 - rho) / rho)
        print(f"  rho {rho:.1f} " + "".join(
            f"{majority(m, 0.7, rho):8.4f}" for m in ms) + f"  limit {lim:.4f}")

    rng = np.random.default_rng(0)              # simulation check
    p, rho, m, n = 0.7, 0.3, 5, 400000
    s = (1 - rho) / rho
    q = rng.beta(p * s, (1 - p) * s, n)         # common difficulty of a case
    right = rng.random((n, m)) < q[:, None]
    print("simulation m=5 rho=0.3:", round(float((right.sum(1) > 2).mean()), 4),
          " formula:", round(float(majority(5, 0.7, 0.3)), 4),
          " correlation of two voters:",
          round(float(np.corrcoef(right[:, 0], right[:, 1])[0, 1]), 3))
    res = {"ms": ms, "indep": {str(p): [float(majority(m, p)) for m in ms]
                               for p in (0.55, 0.6, 0.7, 0.45)},
           "corr": {str(r): [float(majority(m, 0.7, r)) for m in ms]
                    for r in (0.1, 0.3, 0.6)}}
    json.dump(res, open("verify/condorcet.json", "w"))
