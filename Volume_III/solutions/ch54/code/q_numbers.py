# Numbers quoted in the model answers to the questions for discussion (Chapter 54 solutions)
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "1"
import sys, tempfile
sys.path.insert(0, '/home/claude/mlbook3/ch54/code')
import numpy as np, pandas as pd
from scipy.special import gammaln
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from cars import make_cars

print("== Q1: a scaler and a selector fitted before the cross-validation (60 rows, 2000 noise columns)")
n, p, k, reps = 60, 2000, 10, 30
res = {"scaler first": [], "scaler inside": [], "selector first": [], "selector inside": []}; shift = []
for seed in range(reps):
    rng = np.random.RandomState(seed); X = rng.normal(size=(n, p)); y = rng.randint(0, 2, n)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed); clf = LogisticRegression(max_iter=1000)
    res["scaler first"].append(cross_val_score(clf, StandardScaler().fit_transform(X), y, cv=cv).mean())
    res["scaler inside"].append(cross_val_score(make_pipeline(StandardScaler(), clf), X, y, cv=cv).mean())
    res["selector first"].append(cross_val_score(clf, SelectKBest(f_classif, k=k).fit_transform(X, y), y, cv=cv).mean())
    res["selector inside"].append(cross_val_score(make_pipeline(SelectKBest(f_classif, k=k), clf), X, y, cv=cv).mean())
    tr = next(cv.split(X, y))[0]; sd = X.std(0)
    shift.append(np.mean(np.abs(X.mean(0) - X[tr].mean(0)) / sd))
for key, v in res.items(): print(f"{key:16s} {np.mean(v):.3f} +- {np.std(v):.3f}")
print("mean shift of a column mean when the 12 test rows are left out, in sd units: %.3f (theory sqrt(12/(60*48)) = %.3f)" % (np.mean(shift), np.sqrt(12 / (60 * 48))))
print("largest chance correlation sqrt(2 ln(2p)/n) = %.3f" % np.sqrt(2 * np.log(2 * p) / n))
print("ln C(2000,10) = %.1f nats of choice in the selector; the scaler makes none" % (gammaln(p + 1) - gammaln(k + 1) - gammaln(p - k + 1)))

print("\n== Q2: one StandardScaler object in two pipelines")
rng = np.random.RandomState(0); X1 = rng.normal(0, 1, (200, 1)); y1 = (X1[:, 0] > 0).astype(int)
X2 = rng.normal(100, 5, (200, 1)); y2 = (X2[:, 0] > 100).astype(int)
sc = StandardScaler()
A = make_pipeline(sc, LogisticRegression()); B = make_pipeline(sc, LogisticRegression())
A.fit(X1, y1); print("pipeline A fitted on data 1: accuracy %.3f, scaler mean %.2f" % (A.score(X1, y1), sc.mean_[0]))
B.fit(X2, y2); print("pipeline B fitted on data 2: accuracy %.3f, scaler mean %.2f" % (B.score(X2, y2), sc.mean_[0]))
print("pipeline A again on data 1:   accuracy %.3f   (same scaler object: A[0] is B[0] = %s)" % (A.score(X1, y1), A[0] is B[0]))
sc = StandardScaler(); A = make_pipeline(clone(sc), LogisticRegression()).fit(X1, y1); B = make_pipeline(clone(sc), LogisticRegression()).fit(X2, y2)
print("with clone(sc) for each pipeline: A %.3f, B %.3f, means %.2f and %.2f" % (A.score(X1, y1), B.score(X2, y2), A[0].mean_[0], B[0].mean_[0]))
mem = tempfile.mkdtemp(); sc = StandardScaler(); C = make_pipeline(sc, LogisticRegression(), memory=mem).fit(X1, y1)
print("with memory: the pipeline holds a clone, the object passed in is unfitted: has mean_ = %s; pipeline's own scaler has mean_ = %s; same object = %s" % (hasattr(sc, "mean_"), hasattr(C[0], "mean_"), C[0] is sc))

print("\n== Q3: a new column arrives at transform")
df = make_cars(300, 0); X = df.drop(columns="price"); cols = ["colour", "fuel", "age"]
for rem in ("drop", "passthrough"):
    ct = ColumnTransformer([("c", OneHotEncoder(), ["colour", "fuel"]), ("n", StandardScaler(), ["age"])], remainder=rem).fit(X[cols + ["km"]])
    new = X[cols + ["km"]].copy(); new["extra"] = 1.0
    try: out = ct.transform(new); print(f"remainder={rem!r:14}: fitted on 4 columns, new table has 5 -> output shape {out.shape}, no warning")
    except Exception as e: print(f"remainder={rem!r:14}: fitted on 4 columns, new table has 5 -> {type(e).__name__}: {str(e)[:150]}")
    new2 = X[cols + ["km"]].copy(); new2.insert(1, "extra", 1.0)
    try: out = ct.transform(new2); print(f"remainder={rem!r:14}: new column inserted in the middle -> output shape {out.shape}")
    except Exception as e: print(f"remainder={rem!r:14}: new column inserted in the middle -> {type(e).__name__}: {str(e)[:150]}")

for rem in ("drop", "passthrough"):
    ct = ColumnTransformer([("c", OneHotEncoder(), [0, 1]), ("n", StandardScaler(), [2])], remainder=rem).fit(X[cols + ["km"]].to_numpy())
    try: out = ct.transform(np.c_[X[cols + ["km"]].to_numpy(), np.ones(300)]); print(f"remainder={rem!r:14}: same, but plain arrays -> output shape {out.shape}")
    except Exception as e: print(f"remainder={rem!r:14}: same, but plain arrays -> {type(e).__name__}: {str(e)[:130]}")

ct = ColumnTransformer([("c", OneHotEncoder(), ["colour", "fuel"]), ("n", StandardScaler(), ["age"])], remainder="passthrough").fit(X)
print("at fit, remainder='passthrough' on the full 6-column table keeps the other %d columns: output width %d" % (X.shape[1] - 3, ct.transform(X).shape[1]))
ct = ColumnTransformer([("c", OneHotEncoder(), [0, 1]), ("n", StandardScaler(), [2])], remainder="passthrough").fit(X[cols + ["km"]].to_numpy())
try: ct.transform(np.c_[X[cols + ["km"]].to_numpy(), np.ones(300)])
except Exception as e: print("plain arrays, passthrough:", type(e).__name__, str(e)[:100])

print("\n== Q4: totals and typical prices after a log target (sigma = 0.6)")
rng = np.random.RandomState(0); N = 200000; x = rng.uniform(0, 3, (N, 1)); y = np.exp(1 + 0.8 * x[:, 0] + rng.normal(0, 0.6, N))
t = TransformedTargetRegressor(LinearRegression(), func=np.log, inverse_func=np.exp).fit(x, y); med = t.predict(x)
sm = np.exp(np.log(y) - t.regressor_.predict(x)).mean()
print("total of the true prices %.0f; sum of median predictions %.0f (%.1f per cent low); smeared %.0f (%.2f per cent off)" % (y.sum(), med.sum(), 100 * (1 - med.sum() / y.sum()), sm * med.sum(), 100 * (sm * med.sum() / y.sum() - 1)))
print("smearing factor %.4f, exp(sigma^2/2) = %.4f" % (sm, np.exp(0.18)))
print("typical price: share of true prices below the median prediction %.3f, below the smeared one %.3f" % ((y < med).mean(), (y < sm * med).mean()))
print("mean absolute error: median %.3f, smeared %.3f; mean squared error: median %.1f, smeared %.1f" % (np.abs(y - med).mean(), np.abs(y - sm * med).mean(), ((y - med) ** 2).mean(), ((y - sm * med) ** 2).mean()))
