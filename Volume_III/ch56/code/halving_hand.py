# Section 56.4: the schedule of successive halving, by hand and run.
import math
import numpy as np
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import (HalvingGridSearchCV, ParameterGrid,
                                     StratifiedKFold)
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier

eta, n0, R, r0 = 3, 81, 2430, 30        # factor, candidates, rows
rows = []
for k in range(1 + round(math.log(n0, eta))):
    n_k, r_k = math.ceil(n0 / eta**k), r0 * eta**k   # Eq. 56.3-4
    rows.append((n_k, r_k))
    print(f"rung {k}: {n_k:3d} candidates x {r_k:4d} rows"
          f" = {n_k * r_k}")
print("total", sum(a * b for a, b in rows), " full grid", n0 * R)

grid = {"max_depth": [2, 4, 8], "min_samples_leaf": [1, 5, 20],
        "ccp_alpha": [0.0, 0.002, 0.01],
        "max_features": [0.3, 0.6, 1.0]}
assert len(ParameterGrid(grid)) == n0
X, y = make_classification(n_samples=R, n_informative=6,
                           random_state=0)
hs = HalvingGridSearchCV(
    DecisionTreeClassifier(random_state=0), grid, factor=eta,
    cv=StratifiedKFold(5, shuffle=True, random_state=0),
    random_state=0).fit(X, y)
print("sklearn candidates", hs.n_candidates_)
print("sklearn resources ", hs.n_resources_)
assert hs.n_candidates_ == [a for a, _ in rows]
assert hs.n_resources_ == [b for _, b in rows]
