# Impurity importances favour a noise column with many values.
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

rng = np.random.default_rng(0)
n = 1000
s = rng.integers(0, 3, n)                  # informative, 3 values
u = rng.uniform(size=n)                    # noise, 1000 values
v = rng.integers(0, 2, n)                  # noise, 2 values
y = rng.random(n) < 0.2 + 0.3 * s          # P(y = 1) = 0.2, 0.5, 0.8
X = np.c_[s, u, v]
X_tr, X_te, y_tr, y_te = X[:500], X[500:], y[:500], y[500:]

forest = RandomForestClassifier(n_estimators=300, random_state=0)
forest.fit(X_tr, y_tr)
perm = permutation_importance(forest, X_te, y_te, n_repeats=20,
                              random_state=0)
print("feature       values   MDI (train)   permutation (test)")
for j, name in enumerate(["s (signal)", "u (noise)", "v (noise)"]):
    print(f"{name:12s} {len(np.unique(X[:, j])):7d}   "
          f"{forest.feature_importances_[j]:11.3f}   "
          f"{perm.importances_mean[j]:+.3f}")
print("train accuracy:", forest.score(X_tr, y_tr),
      " test accuracy:", round(forest.score(X_te, y_te), 3))
