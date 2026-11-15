# Problem 27B.3: how many trees does early stopping keep?
import numpy as np
from sklearn.datasets import make_friedman1
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

X, y = make_friedman1(n_samples=600, noise=1.0, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5,
                                      random_state=0)
print("seed  trees kept  test R^2")
kept, r2 = [], []
for seed in range(10):
    g = GradientBoostingRegressor(
        n_estimators=1000, learning_rate=0.1, n_iter_no_change=10,
        random_state=seed).fit(Xtr, ytr)
    kept.append(g.n_estimators_)
    r2.append(g.score(Xte, yte))
    print(f"{seed:4d} {kept[-1]:10d}   {r2[-1]:.4f}")
print(f"trees kept: {min(kept)} to {max(kept)}, "
      f"mean {np.mean(kept):.1f}, std {np.std(kept):.1f}")
print(f"test R^2  : {min(r2):.4f} to {max(r2):.4f}, "
      f"std {np.std(r2):.4f}")
print("validation set per fit:", int(0.1 * len(ytr)), "of", len(ytr))
