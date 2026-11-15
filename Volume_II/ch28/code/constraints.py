# Monotonic and interaction constraints (Section 28.8).
import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor as HGB

rng = np.random.default_rng(0)
x = rng.uniform(0, 10, (400, 1))
y = x[:, 0] + rng.normal(0, 2.5, 400)      # rising, very noisy
grid = np.linspace(0, 10, 500).reshape(-1, 1)
for cst in (None, [1]):
    m = HGB(monotonic_cst=cst, random_state=0).fit(x, y)
    steps = np.diff(m.predict(grid))
    n_fall = int((steps < -1e-9).sum())
    print("monotonic_cst", cst, " falling steps:", n_fall)

X = rng.uniform(-1, 1, (3000, 2))
cases = (("additive   ", X[:, 0] + X[:, 1] ** 2),
         ("interacting", X[:, 0] * X[:, 1]))
for name, f in cases:
    t = f + rng.normal(0, 0.1, 3000)
    line = []
    for cst in (None, "no_interactions"):
        m = HGB(interaction_cst=cst, random_state=0)
        m.fit(X[:2000], t[:2000])
        r2 = m.score(X[2000:], t[2000:])
        line.append("%s R2 %.3f" % (cst, r2))
    print(name, " ".join(line))
