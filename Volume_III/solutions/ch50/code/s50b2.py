# Problem 50B.2: leakage against category size
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import TargetEncoder

kf = KFold(5, shuffle=True, random_state=0)
print(f"{'r':>3}{'categories':>11}{'fit,transform':>15}"
      f"{'fit_transform':>15}{'1/sqrt(r)':>11}")
for r in (1, 2, 3, 5, 10, 30):
    k = 1200 // r                          # categories
    rng = np.random.RandomState(0)
    ids = rng.permutation(np.repeat(np.arange(k), r))
    ids = ids.reshape(-1, 1).astype(str)
    y = rng.normal(size=len(ids))          # unrelated to ids
    te = TargetEncoder(smooth=5.0, cv=kf, target_type="continuous")
    leak = te.fit(ids, y).transform(ids)[:, 0]
    cross = te.fit_transform(ids, y)[:, 0]
    corr = lambda z: np.corrcoef(z, y)[0, 1]
    print(f"{r:3d}{k:11d}{corr(leak):15.4f}{corr(cross):15.4f}"
          f"{1/np.sqrt(r):11.3f}")
print("\nstandard error of a correlation of 1200 noise pairs: "
      f"{1/np.sqrt(1200):.3f}")
