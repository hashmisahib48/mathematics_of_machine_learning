# Test RMSE against the number of cities (figure of section 50.11).
import json, os, time
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (KBinsDiscretizer, OneHotEncoder,
                                   OrdinalEncoder, TargetEncoder)
from cars import make_cars

HERE = os.path.dirname(os.path.abspath(__file__))
CAT, NUM = ["colour", "fuel", "trim", "city"], ["age", "km"]
ENC = {"one-hot": lambda: OneHotEncoder(handle_unknown="ignore",
                                        sparse_output=False),
       "ordinal": lambda: OrdinalEncoder(
           handle_unknown="use_encoded_value", unknown_value=-1),
       "target": lambda: TargetEncoder(
           target_type="continuous",
           cv=KFold(5, shuffle=True, random_state=0))}

def pipe(enc, model):
    ridge = model == "ridge"
    num = (KBinsDiscretizer(10, encode="onehot-dense") if ridge
           else "passthrough")
    ct = ColumnTransformer([("cat", ENC[enc](), CAT), ("num", num, NUM)])
    return make_pipeline(ct, Ridge(1.0) if ridge else HGB(random_state=0))

if __name__ == "__main__":
    CITIES = [10, 50, 200, 800, 2000]
    combos = [("ridge", "one-hot"), ("ridge", "target"),
              ("hgb", "one-hot"), ("hgb", "ordinal"), ("hgb", "target")]
    res = {f"{m} {e}": [] for m, e in combos}
    base = []
    t0 = time.time()
    for nc in CITIES:
        acc = {k: [] for k in res}; b = []
        for seed in range(3):
            df = make_cars(4000, seed, n_cities=nc)
            Xtr, Xte, ytr, yte = train_test_split(
                df[CAT + NUM], df.price, random_state=0)
            b.append(rmse(yte, np.full(len(yte), ytr.mean())))
            for m, e in combos:
                p = pipe(e, m).fit(Xtr, ytr)
                acc[f"{m} {e}"].append(rmse(yte, p.predict(Xte)))
        for k in res: res[k].append(float(np.mean(acc[k])))
        base.append(float(np.mean(b)))
        print(nc, {k: round(v[-1], 2) for k, v in res.items()},
              "baseline", round(base[-1], 2), f"{time.time()-t0:.0f}s")
    json.dump(dict(cities=CITIES, rmse=res, baseline=base),
              open(os.path.join(HERE, "..", "verify",
                           "cardinality.json"), "w"))
