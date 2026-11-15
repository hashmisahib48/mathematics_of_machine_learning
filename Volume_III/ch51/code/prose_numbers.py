"""Every number quoted in the prose of Chapter 51 that no listing prints."""
import json
import warnings
import numpy as np
from threadpoolctl import threadpool_limits
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import (SimpleImputer, KNNImputer, IterativeImputer,
                            MissingIndicator)
from sklearn.metrics.pairwise import nan_euclidean_distances
from sklearn.tree import DecisionTreeRegressor
from holes import split, punch, HOLE
from clinic import X as clinic
threadpool_limits(1)
warnings.simplefilter("ignore", ConvergenceWarning)
Xtr, Xte, ytr, yte = split()
N = {}

# complete rows and realised hole rates (12 punchings, 20 % holes)
for kind in ("MCAR", "MAR", "MNAR"):
    f, r = [], []
    for s in range(12):
        A = punch(Xtr, kind, 0.2, s)
        f.append((~np.isnan(A).any(axis=1)).mean())
        r.append(np.isnan(A[:, HOLE]).mean())
    N[f"complete_frac_{kind}"] = round(float(np.mean(f)), 3)
    N[f"hole_rate_{kind}"] = round(float(np.mean(r)), 3)
N["complete_frac_theory"] = round(0.8 ** 4, 4)
N["n_train"] = len(Xtr)

# what each fill does to the variance of a column and to its
# correlation with a complete column (s6), MCAR 20 %
def fills(A, s):
    return {"mean": SimpleImputer().fit_transform(A),
            "knn": KNNImputer().fit_transform(A),
            "iterative": IterativeImputer(random_state=0).fit_transform(A),
            "posterior": IterativeImputer(sample_posterior=True,
                                          random_state=s).fit_transform(A)}
V, C = {}, {}
for s in range(12):
    A = punch(Xtr, "MCAR", 0.2, s)
    for k, F in fills(A, s).items():
        V.setdefault(k, []).append(np.mean(
            [F[:, j].var() / Xtr[:, j].var() for j in HOLE]))
        C.setdefault(k, []).append(np.mean(
            [np.corrcoef(F[:, j], Xtr[:, 9])[0, 1] /
             np.corrcoef(Xtr[:, j], Xtr[:, 9])[0, 1] for j in HOLE]))
N["var_ratio"] = {k: round(float(np.mean(v)), 3) for k, v in V.items()}
N["corr_ratio"] = {k: round(float(np.mean(v)), 3) for k, v in C.items()}
N["corr_ratio_theory"] = round(0.8 ** 0.5, 3)

# IterativeImputer: rounds, and the stopping rule is a matrix norm
A = punch(Xtr, "MCAR", 0.2, 0)
N["n_iter_default"] = int(IterativeImputer(random_state=0).fit(A).n_iter_)
N["n_iter_tol_1e-2"] = int(IterativeImputer(
    random_state=0, tol=1e-2).fit(A).n_iter_)
prev = IterativeImputer(random_state=0, max_iter=0).fit_transform(A)
scale = np.nanmax(np.abs(A)); mat, mx = [], []
for k in range(1, 11):
    cur = IterativeImputer(random_state=0, max_iter=k).fit_transform(A)
    mat.append(float(np.linalg.norm(cur - prev, ord=np.inf)))
    mx.append(float(np.abs(cur - prev).max())); prev = cur
N["change_matrix_inf_norm"] = [round(v, 4) for v in mat]
N["change_max_abs"] = [round(v, 4) for v in mx]
N["scaled_tol_1e-2"] = round(1e-2 * scale, 5)
N["first_round_below_1e-2_matrix"] = 1 + int(np.argmax(
    np.array(mat) < 1e-2 * scale))
N["first_round_below_1e-2_maxabs"] = 1 + int(np.argmax(
    np.array(mx) < 1e-2 * scale))

# a tree and a booster send NaN to the larger child if none were seen
rng = np.random.RandomState(0)
x = rng.uniform(size=(300, 1))
y = np.where(x[:, 0] < 0.3, 5.0, 0.0) + rng.normal(scale=0.1, size=300)
t = DecisionTreeRegressor(max_depth=1).fit(x, y)
N["tree_child_samples"] = t.tree_.n_node_samples[1:].tolist()
N["tree_nan_equals_larger_child"] = bool(
    t.predict([[np.nan]])[0] == t.predict([[1.0]])[0])
x2, y2 = x.copy(), y.copy()
m = rng.uniform(size=300) < 0.2
x2[m], y2[m] = np.nan, 5.0 + rng.normal(scale=0.1, size=m.sum())
t2 = DecisionTreeRegressor(max_depth=1).fit(x2, y2)
N["tree_nan_seen_goes_to_learned_side"] = bool(
    t2.predict([[np.nan]])[0] == t2.predict([[0.0]])[0])
h = HGB(max_iter=1, max_depth=1, learning_rate=1.0, min_samples_leaf=5,
        early_stopping=False).fit(x, y)
N["hgb_nan_equals_larger_child"] = bool(
    h.predict([[np.nan]])[0] == h.predict([[1.0]])[0])

# which estimators accept NaN (tags), and encoders and scalers
from sklearn.utils import get_tags
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                              GradientBoostingRegressor)
from sklearn.tree import ExtraTreeRegressor
from sklearn.preprocessing import (OneHotEncoder, OrdinalEncoder,
                                   StandardScaler)
N["allow_nan"] = {type(e).__name__: bool(get_tags(e).input_tags.allow_nan)
    for e in (DecisionTreeRegressor(), ExtraTreeRegressor(),
              RandomForestRegressor(), ExtraTreesRegressor(), HGB(),
              GradientBoostingRegressor(), Ridge(), LinearRegression(),
              KNeighborsRegressor(), SVR(), StandardScaler())}
cat = np.array([["a"], ["b"], [np.nan], ["a"]], dtype=object)
N["onehot_categories"] = [str(c) for c in
    OneHotEncoder().fit(cat).categories_[0]]
N["ordinal_nan_code"] = str(OrdinalEncoder().fit_transform(cat)[2, 0])
N["scaler_mean_ignores_nan"] = StandardScaler().fit(
    np.array([[1, 10], [2, np.nan], [3, 30.]])).mean_.tolist()

# Example 51.6-1 with weights="distance"
d = nan_euclidean_distances(clinic[1:2], clinic)[0]
w = 1 / d[[4, 3]]
N["hand_distance_weighted_bp"] = round(float(
    w @ clinic[[4, 3], 1] / w.sum()), 3)
N["hand_sklearn_distance_weighted_bp"] = round(float(KNNImputer(
    n_neighbors=2, weights="distance").fit_transform(clinic)[1, 1]), 3)

# complete-data scores of the case study
from sklearn.linear_model import LinearRegression as LR
N["complete_LR"] = round(LR().fit(Xtr, ytr).score(Xte, yte), 3)
N["complete_HGB"] = round(HGB(learning_rate=0.03, max_depth=2,
    max_iter=150, min_samples_leaf=10, random_state=0).fit(
    Xtr, ytr).score(Xte, yte), 3)

# nan_euclidean_distances is nan without a shared coordinate
Z = np.array([[1, np.nan], [np.nan, 2.0]])
N["no_shared_coordinate_distance"] = str(nan_euclidean_distances(Z)[0, 1])
N["indicator_dtype"] = str(MissingIndicator().fit_transform(clinic).dtype)

# spread of the case study
R = json.load(open("../verify/case.json"))
N["case_sd_mcar_0.2"] = {k: round(float(np.std(v, ddof=1)), 3)
                         for k, v in R["MCAR_0.2"].items()}
for kind in ("MCAR", "MAR", "MNAR"):
    d = np.array(R[f"{kind}_0.2"]["iterative"]) - np.array(
        R[f"{kind}_0.2"]["mean"])
    N[f"iter_minus_mean_{kind}"] = [round(float(d.mean()), 3),
        round(float(d.std(ddof=1) / np.sqrt(len(d))), 3)]
d = np.array(R["MNAR_0.2"]["mean+ind"]) - np.array(R["MNAR_0.2"]["mean"])
N["ind_minus_mean_MNAR"] = [round(float(d.mean()), 3),
                            round(float(d.std(ddof=1) / np.sqrt(len(d))), 3)]
d = np.array(R["MCAR_0.2"]["mean+ind"]) - np.array(R["MCAR_0.2"]["mean"])
N["ind_minus_mean_MCAR"] = [round(float(d.mean()), 3),
                            round(float(d.std(ddof=1) / np.sqrt(len(d))), 3)]
for q in (0.1, 0.2, 0.3):
    N[f"table_{q}"] = {kd: {n: round(float(np.mean(v)), 3)
        for n, v in R[f"{kd}_{q}"].items()} for kd in ("MCAR", "MAR", "MNAR")}
json.dump(N, open("../verify/prose_numbers.json", "w"), indent=1)
print(json.dumps(N, indent=1))
