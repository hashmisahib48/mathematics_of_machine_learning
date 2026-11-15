# Section 59.5: house prices, candidates by 5-fold CV.
import numpy as np
from sklearn.compose import (ColumnTransformer,
                             TransformedTargetRegressor as TTR)
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
from sklearn.impute import SimpleImputer as Imp
from sklearn.linear_model import Ridge
from sklearn.model_selection import (KFold, cross_validate,
                                     train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (OneHotEncoder, OrdinalEncoder,
                                   StandardScaler, TargetEncoder)
from house import make_houses, CONDS

df = make_houses().drop_duplicates().drop(columns="agent_fee")
train, test = train_test_split(df, test_size=0.25, random_state=0)
Xtr, ytr = train.drop(columns="price"), train["price"]
kf = KFold(5, shuffle=True, random_state=0)   # for TargetEncoder

def prep():
    fill = lambda: Imp(strategy="median", add_indicator=True)
    order = OrdinalEncoder(categories=[CONDS + [np.nan]],
                           encoded_missing_value=np.nan)
    one = OneHotEncoder(handle_unknown="ignore",
                        sparse_output=False)
    return ColumnTransformer([
        ("num", make_pipeline(fill(), StandardScaler()),
         ["area", "rooms", "age", "dist"]),
        ("cond", make_pipeline(order, fill()), ["condition"]),
        ("cat", one, ["kind", "garden"]),
        ("district", TargetEncoder(cv=kf), ["district"])])

log = lambda m: TTR(m, func=np.log, inverse_func=np.exp)
ridge = lambda: make_pipeline(prep(), Ridge(alpha=1.0))
hgb = lambda: make_pipeline(prep(), HGB(random_state=0))
models = {"mean of prices": DummyRegressor(), "ridge": ridge(),
          "ridge, log price": log(ridge()), "hist. boosting": hgb(),
          "hist. boosting, log": log(hgb())}
if __name__ == "__main__":
    for name, m in models.items():
        r = cross_validate(m, Xtr, ytr, cv=5, scoring={
            "r2": "r2", "mae": "neg_mean_absolute_error"})
        print(f"{name:20s}R2 {r['test_r2'].mean():6.3f}   "
              f"MAE {-r['test_mae'].mean():6.1f}")
