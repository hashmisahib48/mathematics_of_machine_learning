import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (KBinsDiscretizer, OneHotEncoder,
                                   OrdinalEncoder, TargetEncoder)
from cars import make_cars

CAT, NUM = ["colour", "fuel", "trim", "city"], ["age", "km"]
KF = KFold(5, shuffle=True, random_state=0)
ENC = {"one-hot": OneHotEncoder(handle_unknown="ignore",
                                sparse_output=False),
       "ordinal": OrdinalEncoder(handle_unknown="use_encoded_value",
                                 unknown_value=-1),
       "target": TargetEncoder(target_type="continuous", cv=KF)}

def pipe(enc, model):
    ridge = model == "ridge"            # bins let the line bend
    bins = KBinsDiscretizer(10, encode="onehot-dense")
    num = bins if ridge else "passthrough"
    ct = ColumnTransformer([("cat", ENC[enc], CAT), ("num", num, NUM)])
    est = Ridge(1.0) if ridge else HGB(random_state=0)
    return make_pipeline(ct, est)

rows = {}
for seed in range(5):
    df = make_cars(4000, seed)
    X, y = df[CAT + NUM], df.price
    Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=0)
    for model in ("ridge", "hgb"):
        for enc in ENC:
            p = pipe(enc, model).fit(Xtr, ytr)
            rows.setdefault((model, enc), []).append(
                rmse(yte, p.predict(Xte)))
    Xc = X.astype({c: "category" for c in CAT})   # native
    Xtr, Xte, ytr, yte = train_test_split(Xc, y, random_state=0)
    m = HGB(categorical_features="from_dtype", random_state=0)
    rows.setdefault(("hgb", "native"), []).append(
        rmse(yte, m.fit(Xtr, ytr).predict(Xte)))
res = pd.DataFrame(rows).agg(["mean", "std"]).T.round(2)
print(res.rename(columns={"mean": "mean RMSE", "std": "sd"}))
