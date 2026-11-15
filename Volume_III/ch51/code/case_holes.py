import json
import warnings
import numpy as np
from threadpoolctl import threadpool_limits
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import (SimpleImputer, KNNImputer,
                            IterativeImputer)
from sklearn.linear_model import LinearRegression as LR
from sklearn.pipeline import make_pipeline
from holes import split, punch
threadpool_limits(1)                      # tiny data: one thread
warnings.simplefilter("ignore", ConvergenceWarning)   # see 51.10

Xtr, Xte, ytr, yte = split()
hgb = HGB(learning_rate=0.03, max_depth=2, max_iter=150,
          min_samples_leaf=10, random_state=0)
IMP = {"mean": SimpleImputer(),
       "mean+ind": SimpleImputer(add_indicator=True),
       "KNN k=5": KNNImputer(n_neighbors=5),
       "iterative": IterativeImputer(random_state=0)}
def scores(A, B):                         # train on A, test on B
    ok = ~np.isnan(A).any(axis=1)         # delete incomplete rows
    m = make_pipeline(SimpleImputer(), LR()).fit(A[ok], ytr[ok])
    out = {"complete rows": m.score(B, yte)}
    for n, imp in IMP.items():
        m = make_pipeline(clone(imp), LR()).fit(A, ytr)
        out[n] = m.score(B, yte)
    out["native NaN"] = clone(hgb).fit(A, ytr).score(B, yte)
    return out

R = {}
for kind in ("MCAR", "MAR", "MNAR"):
    for q in (0.1, 0.2, 0.3):             # test rows get holes too
        runs = [scores(punch(Xtr, kind, q, s),
                       punch(Xte, kind, q, 100 + s))
                for s in range(12)]
        R[f"{kind}_{q}"] = {n: [r[n] for r in runs]
                            for n in runs[0]}
json.dump(R, open("../verify/case.json", "w"))
print("mean test R^2 over 12 patterns, 20% blank cells")
print(f"{'':15s}   MCAR    MAR   MNAR")
for n in R["MCAR_0.2"]:
    v = [np.mean(R[f"{k}_0.2"][n]) for k in ("MCAR", "MAR", "MNAR")]
    print(f"{n:15s}", "".join(f"{x:7.3f}" for x in v))
