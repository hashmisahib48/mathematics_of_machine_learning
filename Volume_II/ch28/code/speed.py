# Fit time of GradientBoosting* and HistGradientBoosting*.
import json
import time
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.ensemble import HistGradientBoostingClassifier as HGBC
from sklearn.ensemble import HistGradientBoostingRegressor as HGBR
from threadpoolctl import threadpool_limits

ns = [1000, 3000, 10000, 30000, 100000]
rows = {"n": ns, "gbc": [], "hgbc": [], "gbr": [], "hgbr": []}
with threadpool_limits(limits=1):          # one thread, for stability
    for n in ns:
        Xc, yc = make_classification(n, 20, n_informative=8,
                                     random_state=0)
        Xr, yr = make_regression(n, 20, n_informative=8, noise=10,
                                 random_state=0)
        jobs = [("gbc", GBC(random_state=0), Xc, yc, 30000),
                ("hgbc", HGBC(early_stopping=False, random_state=0),
                 Xc, yc, 10**9),
                ("gbr", GBR(random_state=0), Xr, yr, 10000),
                ("hgbr", HGBR(early_stopping=False, random_state=0),
                 Xr, yr, 10**9)]
        for key, est, X, y, nmax in jobs:
            if n > nmax:
                rows[key].append(None)        # too slow to wait for
                continue
            t0 = time.perf_counter()
            est.fit(X, y)
            rows[key].append(round(time.perf_counter() - t0, 2))
        print(n, {k: v[-1] for k, v in rows.items() if k != "n"})
with open("verify/speed.json", "w") as f:
    json.dump(rows, f)
