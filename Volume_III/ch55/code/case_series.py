# Section 55.9: a weekly sales series (invented), 20 draws.
import json, numpy as np
import sklearn.model_selection as ms
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from cvdata import make_sales

rf = RandomForestRegressor(50, random_state=0)
models = {"random forest": rf, "ridge": Ridge(alpha=1.0)}
cvs = {"KFold": ms.KFold(5, shuffle=True, random_state=0),
       "TimeSeriesSplit": ms.TimeSeriesSplit(5, test_size=30)}
cols = list(cvs) + ["next 60 weeks"]
res = {(m, c): [] for m in models for c in cols}
for seed in range(20):
    X, y = make_sales(300, seed)
    Xa, ya, Xf, yf = X[:240], y[:240], X[240:], y[240:]
    for m, est in models.items():
        for c, cv in cvs.items():
            s = ms.cross_val_score(est, Xa, ya, cv=cv,
                scoring="neg_root_mean_squared_error")
            res[(m, c)].append(-s.mean())
        p = est.fit(Xa, ya).predict(Xf)       # the real future
        res[(m, cols[-1])].append(np.sqrt(np.mean((p - yf) ** 2)))
print("RMSE, mean of 20 series (sd in brackets)")
print(" " * 14 + "".join(f"{c:>17s}" for c in cols))
for m in models:
    line = f"{m[:13]:14s}"
    for c in cols:
        v = np.array(res[(m, c)])
        line += f"{v.mean():10.2f} ({v.std(ddof=1):4.2f})"
    print(line)
# (the figures read these numbers)
out = {m + "|" + c: [float(np.mean(res[k])),
       float(np.std(res[k], ddof=1))] for k in res
       for m, c in [k]}
json.dump(out, open("verify/case_series.json", "w"))
