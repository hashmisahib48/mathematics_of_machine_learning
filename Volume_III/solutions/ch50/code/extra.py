# Does the smoothing m remove the leak of Problem 50B.2?  Correlation of the fit-then-transform column with the target
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import TargetEncoder

r, k = 3, 400
rng = np.random.RandomState(0); ids = rng.permutation(np.repeat(np.arange(k), r)).reshape(-1, 1).astype(str); y = rng.normal(size=len(ids))
print("r = 3 rows per category; correlation of the leaky column with y, 1/sqrt(r) = %.3f" % (1 / np.sqrt(r)))
for m in (0.1, 1.0, 5.0, 50.0, 500.0):
    z = TargetEncoder(smooth=m, target_type="continuous", cv=KFold(5, shuffle=True, random_state=0)).fit(ids, y).transform(ids)[:, 0]
    print("  smooth = %6.1f: correlation %.4f   spread of the column %.4f" % (m, np.corrcoef(z, y)[0, 1], z.std()))
