# Numbers quoted in the prose of Chapter 54 (not shown as a listing).
import json, os, tempfile
import numpy as np
from joblib import Memory
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from cars import make_cars

OUT = {}
# --- 1. best of p noise correlations (Eq. 54.3-3)
n = 60
OUT["maxcorr"] = {}
for p in (10, 100, 1000, 2000, 10000):
    rng = np.random.RandomState(1)
    m = []
    for _ in range(100 if p <= 2000 else 20):
        X = rng.normal(size=(n, p)); y = rng.normal(size=n)
        Xc = (X - X.mean(0)) / X.std(0); yc = (y - y.mean()) / y.std()
        m.append(np.abs(Xc.T @ yc / n).max())
    OUT["maxcorr"][p] = {"sim": round(float(np.mean(m)), 3),
                         "formula": round(float(np.sqrt(2 * np.log(2 * p) / n)), 3)}
# --- 2. retransformation bias (Eq. 54.5-1)
rng = np.random.RandomState(0)
N = 20000; x = rng.uniform(0, 3, (N, 1))
OUT["retransform"] = {}
for sig in (0.2, 0.4, 0.6, 0.8):
    y = np.exp(1 + 0.8 * x[:, 0] + rng.normal(0, sig, N))
    t = TransformedTargetRegressor(LinearRegression(), func=np.log,
                                   inverse_func=np.exp).fit(x, y)
    pr = t.predict(x); res = np.log(y) - t.regressor_.predict(x)
    smear = np.exp(res).mean()
    OUT["retransform"][sig] = {
        "ratio_means": round(float(y.sum() / pr.sum()), 4),
        "exp_s2_2": round(float(np.exp(sig**2 / 2)), 4),
        "ratio_median": round(float(np.median(y / pr)), 4),
        "rmse": round(float(np.sqrt(np.mean((y - pr) ** 2))), 3),
        "rmse_smeared": round(float(np.sqrt(np.mean((y - pr * smear) ** 2))), 3),
        "mean_err": round(float(np.mean(y - pr)), 3),
        "mean_err_smeared": round(float(np.mean(y - pr * smear)), 3),
        "smear": round(float(smear), 4)}
# --- 3. memory and in-place fitting
class Counting(BaseEstimator, TransformerMixin):
    calls = 0
    def fit(self, X, y=None):
        type(self).calls += 1; self.mean_ = X.mean(0); return self
    def transform(self, X): return X - self.mean_
X = rng.normal(size=(100, 3)); y = X @ [1, 2, 3.]
for label, mem in (("none", None), ("cache", Memory(tempfile.mkdtemp(), verbose=0))):
    Counting.calls = 0
    first = Counting()
    pipe = Pipeline([("c", first), ("r", Ridge())], memory=mem)
    for a in (0.1, 1.0, 10.0):
        pipe.set_params(r__alpha=a).fit(X, y)
    OUT["memory_" + label] = {"fit_calls": Counting.calls,
                              "original_fitted": hasattr(first, "mean_")}
# --- 4. sparse output of the cars column transformer
df = make_cars(2000, 0); Xc = df.drop(columns="price")
for thr in (0.3, 0.0):
    ct = ColumnTransformer([("cat", OneHotEncoder(), ["colour", "fuel", "trim", "city"]),
                            ("num", StandardScaler(), ["age", "km"])], sparse_threshold=thr)
    Z = ct.fit_transform(Xc)
    nnz = Z.nnz if hasattr(Z, "nnz") else int((Z != 0).sum())
    OUT[f"sparse_thr_{thr}"] = {"type": type(Z).__name__, "shape": list(Z.shape),
                                "density": round(nnz / Z.size if not hasattr(Z, "nnz") else nnz / (Z.shape[0] * Z.shape[1]), 4),
                                "sparse_output_": bool(ct.sparse_output_)}
# --- 5. transformer_weights
from sklearn.datasets import load_breast_cancer
Xb = StandardScaler().fit_transform(load_breast_cancer().data)
u1 = FeatureUnion([("pca", PCA(3, random_state=0))]).fit_transform(Xb)
u2 = FeatureUnion([("pca", PCA(3, random_state=0))], transformer_weights={"pca": 2.0}).fit_transform(Xb)
OUT["weights"] = {"ratio": round(float((u2 / u1)[0, 0]), 4)}
# --- 6. number of nested parameters of the Chapter pipeline
pipe = Pipeline([("prep", ColumnTransformer([("cat", OneHotEncoder(), ["fuel"]),
                                             ("num", StandardScaler(), ["age"])])),
                 ("ridge", Ridge())])
OUT["n_params"] = len(pipe.get_params())
OUT["n_params_top"] = len(pipe.get_params(deep=False))
# --- 7. one scaler object shared by two pipelines (Question 2)
Xs = rng.normal(size=(50, 3)); ys = (Xs[:, 0] > 0).astype(int)
sc = StandardScaler()
a = make_pipeline(sc, LogisticRegression()).fit(Xs, ys)
before = float(sc.mean_[0])
b = make_pipeline(sc, LogisticRegression()).fit(Xs + 5, ys)
OUT["shared"] = {"same_object": a[0] is b[0], "mean_before": round(before, 4),
                 "mean_after": round(float(a[0].mean_[0]), 4)}
for step in ("passthrough", None):
    q = Pipeline([("s", step), ("l", LogisticRegression())]).fit(Xs, ys)
    OUT["step_" + str(step)] = round(float(q.score(Xs, ys)), 4)
# --- 7b. columns are selected by name at transform
Xn = make_cars(300, 0).drop(columns="price")
ctn = ColumnTransformer([("n", StandardScaler(), ["age", "km"])]).fit(Xn)
OUT["by_name"] = {"reversed_columns_same": bool(np.allclose(ctn.transform(Xn), ctn.transform(Xn[Xn.columns[::-1]])))}
try:
    ctn.transform(Xn.drop(columns="km"))
except ValueError as e:
    OUT["by_name"]["missing_column_error"] = str(e)
# --- 8. the case table (case_cars.out)
rows = [l for l in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "verify", "case_cars.out")).read().splitlines()[1:]]
v = {l[:34].strip(): l[34:].split() for l in rows}
tg = float(v["target city, fit then transform"][0]); lk = float(v["target city, fit then transform"][1]); oh = float(v["one-hot city, steps on all rows"][0])
OUT["case_gap"] = {"leak": round(tg - lk, 3), "encoder_gap": round(tg - oh, 3), "share": round((tg - lk) / (tg - oh), 3)}
print(json.dumps(OUT, indent=1))
json.dump(OUT, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "verify", "prose_numbers.json"), "w"), indent=1)
