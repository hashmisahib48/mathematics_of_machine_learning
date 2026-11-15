# Numbers quoted in the prose of Chapter 50.
import json, os
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (KBinsDiscretizer, OneHotEncoder,
                                   TargetEncoder)
from cars import make_cars

HERE = os.path.dirname(os.path.abspath(__file__)); N = {}
df = make_cars(4000, 0)
tr, te = train_test_split(df, random_state=0)
vc = tr.city.value_counts()
N["n_train"], N["n_test"] = len(tr), len(te)
N["cities_train"] = int(len(vc))
N["cities_lt5"] = int((vc < 5).sum()); N["cities_eq1"] = int((vc == 1).sum())
N["test_unseen_city"] = int((~te.city.isin(vc.index)).sum())
N["baseline_rmse"] = float(rmse(te.price, np.full(len(te), tr.price.mean())))
N["mean_price"] = float(df.price.mean())
for c in ("colour", "fuel", "trim", "city"):
    N["levels_" + c] = int(df[c].nunique())

# columns and rank of the full one-hot design
cats = ["colour", "fuel", "trim", "city"]
D = OneHotEncoder(sparse_output=False).fit_transform(tr[cats])
A = np.hstack([np.ones((len(D), 1)), D])
N["onehot_cols"] = D.shape[1]; N["rank_with_intercept"] = int(
    np.linalg.matrix_rank(A))
N["expected_rank"] = 1 + sum(tr[c].nunique() - 1 for c in cats)
for k in (1, 5, 10, 20):
    N[f"cols_minfreq{k}"] = int(OneHotEncoder(
        min_frequency=k, sparse_output=False).fit(tr[["city"]]
        ).transform(tr[["city"]]).shape[1])

# ridge on indicators (centred y, no intercept) = shrunk target encoding
x = tr[["city"]]; yc = tr.price.to_numpy() - tr.price.mean(); a = 3.0
Dc = OneHotEncoder(sparse_output=False).fit_transform(x)
w = Ridge(alpha=a, fit_intercept=False).fit(Dc, yc).coef_
enc = TargetEncoder(smooth=a, target_type="continuous").fit(x, tr.price)
N["ridge_vs_te_maxdiff"] = float(np.abs(
    w + tr.price.mean() - enc.encodings_[0]).max())

# auto smoothing: a city seen once keeps its own price
auto = TargetEncoder(smooth="auto", target_type="continuous").fit(x, tr.price)
single = vc.index[vc == 1]
idx = [list(auto.categories_[0]).index(c) for c in single]
own = np.array([tr.price[tr.city == c].iloc[0] for c in single])
N["single_city_maxdiff"] = float(np.abs(auto.encodings_[0][idx] - own).max())
fixed = TargetEncoder(smooth=5.0, target_type="continuous").fit(x, tr.price)
N["single_city_m5_example"] = [float(own[0]), float(
    fixed.encodings_[0][idx[0]]), float(tr.price.mean())]

# the three binning strategies on km
for s in ("uniform", "quantile", "kmeans"):
    kb = KBinsDiscretizer(5, encode="ordinal", strategy=s).fit(tr[["km"]])
    b = kb.transform(tr[["km"]])[:, 0].astype(int)
    N["km_" + s] = dict(edges=[round(float(e), 1) for e in kb.bin_edges_[0]],
                        counts=np.bincount(b, minlength=5).tolist())
N["km_min_max_median"] = [float(tr.km.min()), float(tr.km.max()),
                          float(tr.km.median())]
a_ = TargetEncoder(target_type="continuous").fit_transform(
    df[["city"]][:500], df.price[:500])
b_ = TargetEncoder(target_type="continuous").fit_transform(
    df[["city"]][:500], df.price[:500])
N["default_cv_runs_differ_by_more_than_1"] = bool(np.abs(a_ - b_).max() > 1)
for k, v in N.items(): print(k, v)
json.dump(N, open(os.path.join(HERE, "..", "verify", "prose_numbers.json"),
                  "w"), indent=1)
