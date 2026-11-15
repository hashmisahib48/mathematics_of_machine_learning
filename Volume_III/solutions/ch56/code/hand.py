# Hand checks for Problems 56A.1 (halving schedule) and 56A.2 (random draws)
import math
import numpy as np
from sklearn.datasets import make_classification
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

# ---- 56A.1: 100 candidates, eta = 3, R = 1800 rows, r_min = 5 * 2 * 2 = 20
n0, eta, R, rmin = 100, 3, 1800, 5 * 2 * 2
print("56A.1  log_3(100) = %.4f   n_req = 1 + floor(log_3 n0) = %d" % (math.log(n0, eta), 1 + math.floor(math.log(n0, eta))))
n_req = 1 + math.floor(math.log(n0, eta))
r0 = max(rmin, R // eta ** (n_req - 1))
print("  floor(R / 3^%d) = %d   r0 = max(%d, %d) = %d" % (n_req - 1, R // eta ** (n_req - 1), rmin, R // eta ** (n_req - 1), r0))
print("  floor(R / r0) = %d   log_3 of it = %.4f   n_poss = %d" % (R // r0, math.log(R // r0, eta), 1 + math.floor(math.log(R // r0, eta))))
n_poss = 1 + math.floor(math.log(R // r0, eta)); nr = min(n_req, n_poss)
rows = [(math.ceil(n0 / eta ** k), min(eta ** k * r0, R)) for k in range(nr)]
for k, (a, b) in enumerate(rows): print("  rung %d: n = ceil(100/3^%d) = %3d (100/%d = %.2f)  r = min(3^%d * 22, 1800) = %4d   cost %5d" % (k, k, a, eta ** k, n0 / eta ** k, k, b, a * b))
tot = sum(a * b for a, b in rows)
print("  total %d   full grid %d   saving %.2f" % (tot, n0 * R, n0 * R / tot))
print("  one rung per candidate-row of the last rung: rows left unused at the top: %d" % (R - rows[-1][1]))
X, y = make_classification(n_samples=R, n_informative=6, random_state=0)
g = {"max_depth": list(range(1, 11)), "min_samples_leaf": [1, 2, 3, 5, 8, 12, 18, 25, 35, 50]}
hs = HalvingGridSearchCV(DecisionTreeClassifier(random_state=0), g, factor=eta,
        cv=StratifiedKFold(5, shuffle=True, random_state=0), random_state=0).fit(X, y)
print("  sklearn candidates", hs.n_candidates_, " resources", hs.n_resources_, " required", hs.n_required_iterations_, " possible", hs.n_possible_iterations_)
assert hs.n_candidates_ == [a for a, _ in rows] and hs.n_resources_ == [b for _, b in rows]

# ---- 56A.2
print("\n56A.2  n >= ln(1-P)/ln(1-q)")
for q, P in ((0.02, 0.99), (0.05, 0.99), (0.10, 0.99)):
    v = math.log(1 - P) / math.log(1 - q)
    print("  q = %.2f  P = %.2f: ln(%.2f) = %.4f  ln(%.2f) = %.5f  ratio %.2f -> n = %d   check 1-(1-q)^n = %.4f" % (q, P, 1 - P, math.log(1 - P), 1 - q, math.log(1 - q), v, math.ceil(v), 1 - (1 - q) ** math.ceil(v)))
print("  with n = 60, q = 0.02: (0.98)^60 = %.4f   P = %.4f   (n = 227 gives %.4f)" % (0.98 ** 60, 1 - 0.98 ** 60, 1 - 0.98 ** 227))
print("  with n = 60 draws and q = 0.05: P = %.4f" % (1 - 0.95 ** 60))
