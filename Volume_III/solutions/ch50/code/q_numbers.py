# Numbers quoted in the model answers to the Questions for discussion of Chapter 50
import os, sys, warnings
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../ch50/code')
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (KBinsDiscretizer, OneHotEncoder, TargetEncoder)
from cars import make_cars

print("== Question 1: the month as a number")
rng = np.random.RandomState(0); m = np.tile(np.arange(1, 13), 10)
T = 11 + 9 * np.cos(2 * np.pi * (m - 7) / 12) + rng.normal(0, 1, 120)       # warm in July, cold in December and January
tr = np.arange(120) < 60
ang = 2 * np.pi * m / 12
feats = {"month as 1..12": m[:, None].astype(float),
         "sin, cos of the month": np.column_stack([np.sin(ang), np.cos(ang)]),
         "one-hot (12 columns)": (m[:, None] == np.arange(1, 13)).astype(float)}
for k, F in feats.items():
    mdl = Ridge(1.0).fit(F[tr], T[tr])
    print(f"  {k:24s} test R2 {r2_score(T[~tr], mdl.predict(F[~tr])):.3f}")
d = lambda a, b: np.hypot(np.sin(2 * np.pi * a / 12) - np.sin(2 * np.pi * b / 12), np.cos(2 * np.pi * a / 12) - np.cos(2 * np.pi * b / 12))
print("  distance December-January: raw %d, sin/cos %.4f;  January-February: raw 1, sin/cos %.4f" % (12 - 1, d(12, 1), d(1, 2)))

print("\n== Question 2: ridge on one-hot columns against a target encoding")
df = make_cars(4000, 0); trn, tst = train_test_split(df, random_state=0)
CAT = ["colour", "fuel", "trim", "city"]; NUM = ["age", "km"]
truth = np.random.RandomState(2024).normal(0, 3.0, 200); cities = np.array([f"C{j:04d}" for j in range(1, 201)])
ct = ColumnTransformer([("cat", OneHotEncoder(sparse_output=False), CAT), ("num", KBinsDiscretizer(10, encode="onehot-dense"), NUM)])
rp = make_pipeline(ct, Ridge(1.0)).fit(trn[CAT + NUM], trn.price)
coef = rp[-1].coef_; ohe = rp[0].named_transformers_["cat"]; cats = list(ohe.categories_[3])
w_city = coef[6 + 4 + 4: 6 + 4 + 4 + 200]; assert cats == list(cities)
te = TargetEncoder(target_type="continuous", cv=KFold(5, shuffle=True, random_state=0)).fit(trn[["city"]], trn.price)
t_city = te.encodings_[0]
n_c = trn.city.value_counts().reindex(cities).values; small = n_c < 5
def stats(v, mask):
    v = v - v[mask].mean(); t = truth - truth[mask].mean()
    return np.corrcoef(v[mask], t[mask])[0, 1]
print("  correlation with the true city effect, all 200 cities:  ridge one-hot weights %.3f   target encoding %.3f" % (stats(w_city, np.ones(200, bool)), stats(t_city, np.ones(200, bool))))
print("  the %d cities with fewer than 5 training cars:           ridge one-hot weights %.3f   target encoding %.3f" % (small.sum(), stats(w_city, small), stats(t_city, small)))

print("\n== Question 3: fit then transform against fit_transform")
X, y = trn[["city"]], trn.price.values; Xt, yt = tst[["city"]], tst.price.values
kf = KFold(5, shuffle=True, random_state=0)
te = TargetEncoder(target_type="continuous", cv=kf)
Zcf = te.fit_transform(X, y)[:, [0]]; Zleak = te.transform(X); Ztest = te.transform(Xt)
for name, Z, yy in (("training, fit then transform", Zleak, y), ("training, fit_transform", Zcf, y), ("test, transform", Ztest, yt)):
    print(f"  {name:30s} sd {Z.std():.3f}  corr with price {np.corrcoef(Z[:, 0], yy)[0, 1]:.3f}")
for name, Ztr in (("fit then transform", Zleak), ("fit_transform", Zcf)):
    lm = LinearRegression().fit(Ztr, y); print(f"  line fitted on the {name:20s}: slope {lm.coef_[0]:.3f}  train R2 {lm.score(Ztr, y):.3f}  test R2 {lm.score(Ztest, yt):.3f}")
print("  cars per city: mean in a training fold of 2400 = %.1f, in the full table 3000 = %.1f" % (2400 / 200, 3000 / 200))

print("\n== Question 4: quantile bins")
rng = np.random.RandomState(0); x = np.where(rng.rand(1000) < 0.5, 0.0, rng.lognormal(1, 1, 1000)).reshape(-1, 1)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always"); kb = KBinsDiscretizer(5, encode="ordinal", strategy="quantile").fit(x)
print("  half of the values are zero; edges requested 5 bins:", kb.bin_edges_[0].round(3), " n_bins_ =", kb.n_bins_[0])
print("  warning:", [str(i.message)[:80] for i in w])
print("  values equal to zero:", int((x == 0).sum()), "of", len(x))
print("  counts per bin:", np.bincount(kb.transform(x)[:, 0].astype(int)).tolist(), " value 0 is in bin", int(kb.transform(np.array([[0.0]]))[0, 0]))
nz = x[x > 0].reshape(-1, 1); kb2 = KBinsDiscretizer(5, encode="ordinal", strategy="quantile").fit(nz)
print("  bins fitted on the non-zero part only:", kb2.bin_edges_[0].round(2))
km = trn[["km"]]; kq = KBinsDiscretizer(5, encode="ordinal", strategy="quantile").fit(km)
print("  km of the cars, quantile edges:", kq.bin_edges_[0].round(1), " width of the last bin %.1f, of the first %.1f" % (np.diff(kq.bin_edges_[0])[-1], np.diff(kq.bin_edges_[0])[0]))
