# Numbers quoted in the prose of Chapter 58 (written to verify/prose_numbers.json).
import json, warnings
import numpy as np
from bcdata import load
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import (FixedThresholdClassifier, StratifiedKFold, TunedThresholdClassifierCV, check_cv)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
Xa, Xb, ya, yb, names = load(); names = [str(n) for n in names]
ym, yt = 1 - ya, 1 - yb
N = {}
model = make_pipeline(StandardScaler(), LogisticRegression())
def cost(y, pred, c_fn, c_fp, pos_label):
    miss = (y == pos_label) & (pred != pos_label); alarm = (y != pos_label) & (pred == pos_label)
    return (c_fn * miss.sum() + c_fp * alarm.sum()) / len(y)
c10 = make_scorer(cost, greater_is_better=False, c_fn=10, c_fp=1, pos_label=0)

# API facts stated in the text
assert str(check_cv(None, ya, classifier=True)).startswith("StratifiedKFold(n_splits=5")
t = TunedThresholdClassifierCV(model).fit(Xa, ya)
N["default_scoring"] = t.scoring; N["has_cv_results_without_store"] = hasattr(t, "cv_results_")
N["default_tuned_threshold"] = round(float(t.best_threshold_), 4); N["default_tuned_score"] = round(float(t.best_score_), 4)
tc = TunedThresholdClassifierCV(model, scoring=c10, store_cv_results=True).fit(Xa, ya)
assert np.allclose(tc.predict_proba(Xb), tc.estimator_.predict_proba(Xb))   # only predict changes
yes = tc.estimator_.predict_proba(Xb)[:, 0] >= tc.best_threshold_
assert ((tc.predict(Xb) == 0) == yes).all()                                  # '>=' rule
fx = FixedThresholdClassifier(model, threshold="auto").fit(Xa, ya)
assert (fx.predict(Xb) == model.fit(Xa, ya).predict(Xb)).all()
wrong = FixedThresholdClassifier(model, threshold=1 / 11).fit(Xa, ya)
N["no_pos_label_flagged_malignant"] = int((wrong.predict(Xb) == 0).sum())
N["t_star_10"] = round(1 / 11, 4); N["logit_t_star_10"] = round(float(np.log((1 / 11) / (10 / 11))), 4)
N["logit_t_star_19"] = round(float(np.log(1 / 19)), 4)

# the library on the five rows of Example 58.5-2 (a stump that asks only about x1)
from sklearn.tree import DecisionTreeClassifier
x1 = np.array([1., 2, 3, 4, 5]); yy = np.array([0, 0, 1, 0, 1]); X5 = np.c_[x1, [3., 1, 4, 1, 5]]
stump = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X5, (x1 >= 3).astype(int))
pi5 = permutation_importance(stump, X5, yy, n_repeats=2000, random_state=0)
N["five_rows_library_importance"] = np.round(pi5.importances_mean, 3).tolist()

# the knob 'thresholds'
N["thresholds_knob"] = {}
for k in (5, 20, 100, 1000):
    tk = TunedThresholdClassifierCV(model, scoring=c10, thresholds=k).fit(Xa, ya)
    N["thresholds_knob"][k] = [round(float(tk.best_threshold_), 4), round(float(tk.best_score_), 4)]

# recursion versus brute partial dependence of the boosting model
gb = GBC(random_state=0).fit(Xa, ym); j = 27
rec = partial_dependence(gb, Xb, [j], grid_resolution=5)               # method='auto'
bru = partial_dependence(gb, Xb, [j], grid_resolution=5, method="brute", response_method="decision_function")
brp = partial_dependence(gb, Xb, [j], grid_resolution=5, method="brute", response_method="predict_proba")
N["pd_auto_recursion_log_odds"] = np.round(rec["average"][0], 3).tolist()
N["pd_brute_log_odds"] = np.round(bru["average"][0], 3).tolist()
N["pd_brute_proba"] = np.round(brp["average"][0], 3).tolist()
N["pd_grid"] = np.round(rec["grid_values"][0], 4).tolist()
# the grid uses 5th and 95th percentiles; np.percentile would give a slightly different end point
N["grid_ends_mquantiles"] = [round(float(v), 5) for v in (rec["grid_values"][0][0], rec["grid_values"][0][-1])]
N["grid_ends_np_percentile"] = [round(float(v), 5) for v in np.percentile(Xb[:, j], [5, 95])]
N["gb_test_acc"] = round(float(gb.score(Xb, yt)), 4)
N["impurity_top3"] = [(names[k], round(float(gb.feature_importances_[k]), 3)) for k in np.argsort(-gb.feature_importances_)[:3]]

# the spread of one importance over the seed of the shuffles
N["repeats_sd"] = {}
jj = names.index("worst texture")
for R in (1, 5, 30):
    v = [permutation_importance(gb, Xb, yt, scoring="neg_log_loss", n_repeats=R, random_state=s).importances_mean[jj] for s in range(10)]
    N["repeats_sd"][R] = [round(float(np.mean(v)), 4), round(float(np.std(v)), 4)]
N["corr_mean_radius_perimeter"] = round(float(np.corrcoef(Xb[:, names.index("mean radius")], Xb[:, names.index("mean perimeter")])[0, 1]), 3)
json.dump(N, open("verify/prose_numbers.json", "w"), indent=1)
print(json.dumps(N, indent=1))
