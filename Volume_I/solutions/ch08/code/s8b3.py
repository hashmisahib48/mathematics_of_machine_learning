# Problem 8B.3: twenty splits of the diabetes data (ten features)
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split

X, y = load_diabetes(return_X_y=True)
r2 = []
for seed in range(20):
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=seed)
    r2.append(LinearRegression().fit(Xtr, ytr).score(Xte, yte))
r2 = np.array(r2)
print("test patients per split:", len(yte))
print("R2 by seed:", r2.round(3))
print(f"min  {r2.min():.3f} (seed {r2.argmin()})")
print(f"max  {r2.max():.3f} (seed {r2.argmax()})")
print(f"mean {r2.mean():.3f}   sd {r2.std(ddof=1):.3f}")
below = int((r2[0] < r2[1:]).sum())
print("seed 0 is below", below, "of the other 19")
gap = (r2.mean() - r2[0]) / r2.std(ddof=1)
print("seed 0 is %.2f sd below the mean" % gap)
print("splits within one sd of the mean:",
      int((abs(r2 - r2.mean()) <= r2.std(ddof=1)).sum()), "of 20")

cv = cross_val_score(LinearRegression(), X, y, cv=5)
print(f"5-fold, all 442: mean {cv.mean():.3f} "
      f"sd {cv.std(ddof=1):.3f}")
