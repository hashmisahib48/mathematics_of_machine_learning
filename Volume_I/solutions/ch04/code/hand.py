# Hand checks for Problems 4A.1 - 4A.4 (every number quoted in the solutions)
import numpy as np
from scipy import stats
np.set_printoptions(precision=4, suppress=True)

print("== 4A.1  3 red, 2 blue, two draws without replacement")
from itertools import permutations
balls = list("RRRBB"); pairs = list(permutations(range(5), 2))
both = sum(balls[i] == "R" and balls[j] == "R" for i, j in pairs)
print("ordered pairs:", len(pairs), " both red:", both, " P =", both / len(pairs))
print("P(R1) =", 3 / 5, " P(R2|R1) =", 2 / 4, " P(R1,R2) =", 3 / 5 * 2 / 4)
print("P(B1) =", 2 / 5, " P(R2|B1) =", 3 / 4, " P(R2,B1) =", 2 / 5 * 3 / 4)
print("P(R2) =", 3 / 5 * 2 / 4 + 2 / 5 * 3 / 4)
r2 = sum(balls[j] == "R" for i, j in pairs) / len(pairs); print("by listing all 20 ordered pairs P(R2) =", r2)

print("\n== 4A.2  2 % prevalence, sensitivity 0.95, false-positive rate 0.10")
p, se, fp = 0.02, 0.95, 0.10
tp, fa = se * p, fp * (1 - p); print("0.95*0.02 =", tp, " 0.10*0.98 =", fa, " P(+) =", tp + fa)
print("P(D|+) =", tp / (tp + fa))
print("per 1000: affected", 1000 * p, " unaffected", 1000 * (1 - p), " TP", 1000 * tp, " FP", 1000 * fa, " positives", 1000 * (tp + fa))
print("19/117 =", 19 / 117, " negatives:", 1000 - 117, " missed:", 1000 * p * (1 - se), " P(D|-) =", p * (1 - se) / (1 - tp - fa))

print("\n== 4A.3  points (1,2) (2,1) (3,4) (4,3)")
x = np.array([1., 2, 3, 4]); y = np.array([2., 1, 4, 3]); dx, dy = x - x.mean(), y - y.mean()
print("means:", x.mean(), y.mean(), " dx:", dx, " dy:", dy)
print("dx*dy:", dx * dy, " dx^2:", dx ** 2, " dy^2:", dy ** 2)
sxy, sxx, syy = (dx * dy).sum(), (dx ** 2).sum(), (dy ** 2).sum(); print("Sxy, Sxx, Syy =", sxy, sxx, syy)
print("cov =", sxy / 3, " var x =", sxx / 3, " var y =", syy / 3, " r =", sxy / np.sqrt(sxx * syy), " np.cov:", np.cov(x, y)[0, 1], " corrcoef:", np.corrcoef(x, y)[0, 1])
print("covariance with divisor n:", sxy / 4, " (the correlation does not depend on the divisor)")

print("\n== 4A.4  heights N(170, 8^2)")
z = (180 - 170) / 8; print("z =", z, " Phi(z) =", stats.norm.cdf(z), " P(X>180) =", 1 - stats.norm.cdf(z), " sf:", stats.norm.sf(z))
print("z_0.9 =", stats.norm.ppf(0.9), " height =", 170 + 8 * stats.norm.ppf(0.9), " ppf with loc, scale:", stats.norm.ppf(0.9, 170, 8))
print("check: P(X <= 180.25) =", stats.norm.cdf(180.25, 170, 8))
print("mixed divisors (Sxy/n over the variances with n-1): r =", (sxy / 4) / (sxx / 3), "  Var with divisor n:", sxx / 4)
