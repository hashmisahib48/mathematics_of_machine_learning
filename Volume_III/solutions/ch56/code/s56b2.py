# Problem 56B.2: halving with the number of trees as the resource
import math
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import (HalvingGridSearchCV,
                                     train_test_split)

X, y = load_breast_cancer(return_X_y=True)
Xa, Xt, ya, yt = train_test_split(X, y, test_size=0.25,
                                  random_state=0, stratify=y)
grid = {"max_depth": [2, 4, 8, None],
        "max_features": ["sqrt", 0.5, 1.0],
        "min_samples_leaf": [1, 3, 9]}
hs = HalvingGridSearchCV(RandomForestClassifier(random_state=0), grid,
        resource="n_estimators", max_resources=81, min_resources=3,
        factor=3, cv=5, random_state=0).fit(Xa, ya)

n, r = hs.n_candidates_, hs.n_resources_
print("candidates per rung:", n)
print("trees per rung     :", r)
print("required %d, possible %d rungs"
      % (hs.n_required_iterations_, hs.n_possible_iterations_))
print("Eq. 56.3-4: n_k =", [math.ceil(36 / 3**k) for k in range(4)],
      " r_k =", [min(3**k * 3, 81) for k in range(4)])
tot = sum(a * b for a, b in zip(n, r))
print("trees per fold: %d against 36 * 81 = %d (saving %.1f)"
      % (tot, 36 * 81, 36 * 81 / tot))
print("winner:", hs.best_params_)
print("five-fold %.4f (last rung)  test %.4f"
      % (hs.best_score_, hs.score(Xt, yt)))
