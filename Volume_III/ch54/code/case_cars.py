import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import TargetEncoder
from cars import make_cars

def holes(n, seed):                     # blank some cells at random
    df, r = make_cars(n, seed), np.random.RandomState(100 + seed)
    df["fuel"] = df.fuel.astype(object)
    for col, share in (("age", .05), ("km", .08), ("fuel", .05)):
        df.loc[r.rand(n) < share, col] = np.nan
    return df.drop(columns="price"), df.price

Xtr, ytr = holes(2000, 0)               # build and cross-validate
Xte, yte = holes(3000, 1)               # fresh cars, used once
kf = KFold(5, shuffle=True, random_state=0)

def prep(city):                    # one branch per kind of column
    num = make_pipeline(SimpleImputer(strategy="median"),
                        StandardScaler())
    cat = make_pipeline(SimpleImputer(strategy="most_frequent"),
                        OneHotEncoder(handle_unknown="ignore"))
    return ColumnTransformer([
        ("num", num, ["age", "km"]),
        ("cat", cat, ["colour", "fuel", "trim"]),
        ("city", city, ["city"])])

onehot = lambda: OneHotEncoder(handle_unknown="ignore")
target = lambda: TargetEncoder(target_type="continuous", cv=kf)
logged = lambda: TransformedTargetRegressor(
    Ridge(), func=np.log, inverse_func=np.exp)
NEG = "neg_root_mean_squared_error"
cv = lambda est, X: -cross_val_score(est, X, ytr, cv=kf,
                                     scoring=NEG).mean()

def row(label, city, leaky, model=Ridge):
    pipe = make_pipeline(prep(city()), model())
    fresh = rmse(yte, pipe.fit(Xtr, ytr).predict(Xte))
    lk = f"{cv(model(), leaky(city())):.3f}" if leaky else "-"
    print(f"{label:34s}{cv(pipe, Xtr):7.3f}{lk:>7s}{fresh:7.3f}")

fit_tr = lambda c: prep(c).fit(Xtr, ytr).transform(Xtr)
fit_ft = lambda c: prep(c).fit_transform(Xtr, ytr)   # cross-fitted
print(f"{'':34s}{'honest':>7s}{'leaky':>7s}{'fresh':>7s}")
row("one-hot city, steps on all rows", onehot, fit_tr)
row("target city, fit then transform", target, fit_tr)
row("target city, fit_transform", target, fit_ft)
row("one-hot city, log target", onehot, None, logged)
