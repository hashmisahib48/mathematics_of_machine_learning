# Question 3 of Chapter 55: blocks of a series
import sys, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../ch55/code')
from cvdata import make_patients, make_sales
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import (KFold, GroupKFold, TimeSeriesSplit, RepeatedKFold, cross_val_score)

# ---- Question 3: blocks of a series
print("\nQ3  (weekly sales, 10 series, RMSE; first 240 weeks are the data)")
cvs = {"TimeSeriesSplit(5, test_size=30)": TimeSeriesSplit(5, test_size=30), "KFold(5) in order": KFold(5), "KFold(5) shuffled": KFold(5, shuffle=True, random_state=0)}
mods = {"forest": RandomForestRegressor(20, random_state=0), "ridge": Ridge(1.0)}
res = {(m, c): [] for m in mods for c in list(cvs) + ["next 60 weeks"]}
for seed in range(10):
    X, y = make_sales(300, seed); Xa, ya, Xf, yf = X[:240], y[:240], X[240:], y[240:]
    for m, est in mods.items():
        for c, cv in cvs.items():
            res[(m, c)].append(-cross_val_score(est, Xa, ya, cv=cv, scoring="neg_root_mean_squared_error").mean())
        res[(m, "next 60 weeks")].append(np.sqrt(np.mean((est.fit(Xa, ya).predict(Xf) - yf)**2)))
for m in mods: print("  %-7s" % m + "".join("  %s %.2f" % (c, np.mean(res[(m, c)])) for c in list(cvs) + ["next 60 weeks"]))
print("  training sizes of TimeSeriesSplit(5, test_size=30) on 240 rows:", [len(a) for a, _ in TimeSeriesSplit(5, test_size=30).split(np.zeros((240, 1)))])
print("  rows ever tested:", sorted(set(np.concatenate([b for _, b in TimeSeriesSplit(5, test_size=30).split(np.zeros((240, 1)))])))[0], "to 239")

