# Numbers quoted in the prose of Chapter 53 (written to verify/).
import json
import numpy as np
from scipy.special import digamma
from scipy.stats import f as f_dist
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    SelectFdr, SelectFpr, SelectFromModel, SelectFwe, chi2, f_classif,
    f_regression, mutual_info_classif, r_regression)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from bcdata import load

N = {}
Xa, Xb, ya, yb, names = load()
nm = [str(a) for a in names]
# correlation
R = np.corrcoef(Xa, rowvar=False)
j1, j2 = nm.index("mean radius"), nm.index("mean perimeter")
N["r_radius_perimeter"] = R[j1, j2]
j3 = nm.index("mean area")
N["r_radius_area"] = R[j1, j3]
N["std_min_max"] = [Xa.std(0).min(), Xa.std(0).max()]
# F equals t^2 and f_regression on 0/1 target
F, p = f_classif(Xa, ya)
F2, p2 = f_regression(Xa, ya)
N["f_equals_f_regression_binary"] = bool(np.allclose(F, F2))
N["fmax"] = [nm[int(F.argmax())], F.max()]
N["fmin"] = [nm[int(F.argmin())], F.min()]
N["F_mean_fractal_dimension"] = F[nm.index("mean fractal dimension")]
N["p_mean_fractal_dimension"] = p[nm.index("mean fractal dimension")]
N["f_crit_1_424_05"] = f_dist.isf(0.05, 1, len(ya) - 2)
N["n_p_below_05"] = int((p < 0.05).sum())
N["not_significant"] = [nm[j] for j in np.flatnonzero(p >= 0.05)]
N["n_p_below_1e-3"] = int((p < 1e-3).sum())
# chi2 depends on the unit; min-max scaled data give another ranking
c_raw, _ = chi2(Xa, ya)
c_mm, _ = chi2(MinMaxScaler().fit_transform(Xa), ya)
N["chi2_raw_top3"] = [nm[j] for j in np.argsort(-c_raw)[:3]]
N["chi2_mm_top3"] = [nm[j] for j in np.argsort(-c_mm)[:3]]
N["chi2_overlap_top5"] = len(set(np.argsort(-c_raw)[:5])
                             & set(np.argsort(-c_mm)[:5]))
# mutual information, and the formula of the source for one column
mi = mutual_info_classif(Xa, ya, random_state=0)
j = nm.index("worst perimeter")
n, k = len(ya), 3
Xs = Xa / Xa.std(0)                       # scale, then tiny noise
E = np.random.RandomState(0).standard_normal(Xa.shape)
x = Xs[:, j] + 1e-10 * max(1, np.abs(Xs[:, j]).mean()) * E[:, j]
rad, cnt = np.empty(n), np.empty(n)
for c in (0, 1):
    m = ya == c
    nn = NearestNeighbors(n_neighbors=k).fit(x[m, None])
    rad[m] = np.nextafter(nn.kneighbors()[0][:, -1], 0)  # self excluded
    cnt[m] = m.sum()
mm = np.array([(np.abs(x - xi) <= r).sum() for xi, r in zip(x, rad)])
N["mi_formula_worst_perimeter"] = (digamma(n) + digamma(k)
    - digamma(cnt).mean() - digamma(mm).mean())
N["mi_sklearn_worst_perimeter"] = mi[j]
# forest importances and the threshold rule
rf = RandomForestClassifier(100, random_state=0).fit(Xa, ya)
imp = rf.feature_importances_
N["rf_threshold_counts"] = {t: int(SelectFromModel(rf, prefit=True,
    threshold=t).get_support().sum())
    for t in ("mean", "median", "1.25*median", "0.5*mean")}
N["rf_mean_importance"] = imp.mean()
# regression filter on diabetes
Xd, yd = load_diabetes(return_X_y=True)
Fd, pd_ = f_regression(Xd, yd)
rd = r_regression(Xd, yd)
nd = len(yd)
N["diabetes_F_top3"] = [(str(load_diabetes().feature_names[k]), Fd[k],
                         rd[k]) for k in np.argsort(-Fd)[:3]]
N["diabetes_F_from_r"] = max(abs(Fd - rd**2 / (1 - rd**2) * (nd - 2)))
# many tests: 20 shifted + 180 null features, 100 samples
rng = np.random.RandomState(0)
res = {a: [] for a in ("fpr", "fwe", "fdr")}
for rep in range(200):
    y = np.arange(100) % 2
    X = rng.standard_normal((100, 200))
    X[:, :20] += 0.6 * y[:, None]
    for a, S in (("fpr", SelectFpr), ("fwe", SelectFwe),
                 ("fdr", SelectFdr)):
        s = S(alpha=0.05).fit(X, y).get_support()
        res[a].append((s[:20].sum(), s[20:].sum()))
N["multitest"] = {}
for a, v in res.items():
    v = np.array(v)
    N["multitest"][a] = [v[:, 0].mean(), v[:, 1].mean(),
        float((v[:, 1] / np.maximum(1, v.sum(1))).mean())]
N["expected_false_fpr"] = 180 * 0.05
N["p_any_false_fpr"] = 1 - 0.95 ** 180
N["sfs_fits_5of30"] = 5 * sum(30 - i for i in range(5))
N["sfs_fits_30"] = 5 * sum(30 - i for i in range(30))
N["rfe_fits_30_to_5"] = 26
def ser(o):
    if isinstance(o, dict): return {str(a): ser(b) for a, b in o.items()}
    if isinstance(o, (list, tuple)): return [ser(b) for b in o]
    if isinstance(o, (np.floating, float)): return round(float(o), 5)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, np.bool_): return bool(o)
    return o
N = ser(N)
json.dump(N, open("verify/prose_numbers.json", "w"), indent=1)
for a, b in N.items():
    print(a, b)
