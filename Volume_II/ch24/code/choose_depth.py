# Validation curves for max_depth and min_samples_leaf (make_moons).
import json
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve
from sklearn.tree import DecisionTreeClassifier

X, y = make_moons(n_samples=300, noise=0.3, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.33, random_state=0, stratify=y)
tree = DecisionTreeClassifier(random_state=0)
grids = {"max_depth": list(range(1, 13)),
         "min_samples_leaf": [1, 2, 3, 5, 8, 12, 20, 30, 50]}
out = {}
for name, grid in grids.items():
    tr, va = validation_curve(tree, X_tr, y_tr, param_name=name,
                              param_range=grid, cv=5)
    tr_m, va_m = tr.mean(1), va.mean(1)
    out[name] = dict(grid=grid, train=tr_m.tolist(),
                     valid=va_m.tolist(), sd=va.std(1).tolist())
    k = int(np.argmax(va_m))
    print(f"{name}: best {grid[k]}, 5-fold {va_m[k]:.4f},",
          f"train {tr_m[k]:.4f}")
    print("  validation:", np.round(va_m, 3))
with open("verify/choose_depth.json", "w") as f:
    json.dump(out, f)
