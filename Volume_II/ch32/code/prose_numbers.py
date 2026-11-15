# Not printed: numbers quoted in the text of Chapter 32.
import json
import warnings
import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.isotonic import IsotonicRegression, isotonic_regression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

R = {}
rng = np.random.default_rng(0)
Z = rng.normal(size=(25000, 3))
X = np.hstack([Z] + [Z + 0.3 * rng.normal(size=Z.shape)
                     for _ in range(3)])
p_true = 1 / (1 + np.exp(-Z @ [1.6, 1.28, 0.96]))
y = (rng.random(25000) < p_true).astype(int)
X_tr, y_tr = X[:5000], y[:5000]
X_ca, y_ca = X[5000:15000], y[5000:15000]
X_te, y_te = X[15000:], y[15000:]

# 1. y_min / y_max by clipping = the bounded problem (toy A)
yy = np.array([2, 5, 3, 4, 1, 6, 5, 7.0])
clipped = IsotonicRegression(y_min=3, y_max=6).fit(np.arange(1, 9.), yy)
cons = [{"type": "ineq", "fun": lambda f, i=i: f[i + 1] - f[i]}
        for i in range(7)]
q = minimize(lambda f: ((f - yy) ** 2).sum(), np.full(8, 4.5),
             bounds=[(3, 6)] * 8, constraints=cons, method="SLSQP",
             options={"ftol": 1e-14, "maxiter": 500})
R["bounded_qp_vs_clip"] = float(np.abs(q.x - clipped.predict(
    np.arange(1, 9.))).max())

# 2. prefix property (KKT) on random data
w = rng.integers(1, 5, 60).astype(float)
yr = np.cumsum(rng.normal(0, 1, 60)) * 0.2 + rng.normal(0, 1, 60)
fit = isotonic_regression(yr, sample_weight=w)
bad, nblk = 0.0, 0
s = 0
while s < 60:
    e = s
    while e + 1 < 60 and fit[e + 1] == fit[s]:
        e += 1
    nblk += 1
    cw = np.cumsum(w[s:e + 1])
    pre = np.cumsum(w[s:e + 1] * yr[s:e + 1]) / cw
    bad = min(bad, float((pre - fit[s]).min()))
    s = e + 1
R["prefix_min_slack"] = bad
R["blocks_random"] = nblk
R["weighted_mean_kept"] = float(np.abs((w * fit).sum() - (w * yr).sum()))
# invariance to increasing transformation of x
xs = np.sort(rng.uniform(1, 9, 40))
yx = np.sqrt(xs) + rng.normal(0, 0.5, 40)
a = IsotonicRegression().fit(xs, yx).predict(xs)
b = IsotonicRegression().fit(np.log(xs), yx).predict(np.log(xs))
R["transform_invariance"] = float(np.abs(a - b).max())
# Spearman rho = 0 counts as increasing
xz = np.array([1, 2, 3, 4.0])
yz = np.array([1, 3, 3, 1.0])
with warnings.catch_warnings(record=True) as wl:
    warnings.simplefilter("always")
    inc = bool(IsotonicRegression(increasing="auto")
               .fit(xz, yz).increasing_)
R["spearman_zero"] = [float(spearmanr(xz, yz)[0]), inc,
                      len(wl), [str(w.category.__name__) for w in wl]]

# 3. calibration_curve edge rule: a value on an edge is in the lower bin
t, p_ = calibration_curve([0, 1, 0, 1], [0.2, 0.2, 0.4, 0.6],
                          n_bins=5)
R["edge_rule"] = [t.tolist(), p_.tolist()]

# 4. calibrator input for predict_proba-only estimators
nb = GaussianNB().fit(X_tr, y_tr)
sk = CalibratedClassifierCV(FrozenEstimator(nb)).fit(X_ca, y_ca)
cal = sk.calibrated_classifiers_[0].calibrators[0]
f_prob = nb.predict_proba(X_ca)[:, 1]
n1 = y_ca.sum()
n0 = len(y_ca) - n1
tt = np.where(y_ca == 1, (n1 + 1) / (n1 + 2), 1 / (n0 + 2))


def loss(ab, f):
    z = ab[0] * f + ab[1]
    return (tt * np.logaddexp(0, z) + (1 - tt) * np.logaddexp(0, -z)).sum()


mine = minimize(loss, [0, np.log((n0 + 1) / (n1 + 1))],
                args=(f_prob,), method="BFGS", tol=1e-10).x
lg = np.log(f_prob + 1e-12) - np.log(1 - f_prob + 1e-12)
mine_logit = minimize(loss, [0, np.log((n0 + 1) / (n1 + 1))],
                      args=(lg,), method="BFGS", tol=1e-10).x
R["platt_on_prob_NB"] = {"sklearn_AB": [float(cal.a_), float(cal.b_)],
                         "mine_on_probability": mine.tolist(),
                         "mine_on_logit": mine_logit.tolist()}

# 5. AUC and ties
svm = LinearSVC().fit(X_tr, y_tr)
f_te = svm.decision_function(X_te)
out = {"raw": roc_auc_score(y_te, f_te)}
for m in ("sigmoid", "isotonic", "temperature"):
    c = CalibratedClassifierCV(FrozenEstimator(svm), method=m)
    c.fit(X_ca, y_ca)
    pr = c.predict_proba(X_te)[:, 1]
    out[m] = roc_auc_score(y_te, pr)
    out[m + "_distinct"] = int(len(np.unique(pr)))
    out[m + "_zero_one"] = int(((pr == 0) | (pr == 1)).sum())
R["auc_linsvc"] = out
base = y_ca.mean()
const = np.full(len(y_te), base)
R["constant"] = {"prob": float(base), "brier": brier_score_loss(y_te, const),
                 "unc": float(y_te.mean() * (1 - y_te.mean())),
                 "auc": roc_auc_score(y_te, const),
                 "logloss": log_loss(y_te, const)}

# 6. multiclass: OvR + renormalisation, temperature keeps the argmax
Xw, yw = load_wine(return_X_y=True)
mc = make_pipeline(StandardScaler(), LinearSVC())
cw = CalibratedClassifierCV(FrozenEstimator(mc.fit(Xw, yw)),
                            method="sigmoid").fit(Xw, yw)
cc = cw.calibrated_classifiers_[0]
dec = mc.decision_function(Xw)
raw = np.column_stack([cc.calibrators[k].predict(dec[:, k])
                       for k in range(3)])
mine = raw / raw.sum(1, keepdims=True)
R["wine_ovr"] = {"max_diff": float(np.abs(mine - cw.predict_proba(Xw)).max()),
                 "row_sum_min": float(raw.sum(1).min()),
                 "row_sum_max": float(raw.sum(1).max()),
                 "argmax_agree_sigmoid": float((cw.predict(Xw) == mc.predict(Xw)).mean())}
ct = CalibratedClassifierCV(FrozenEstimator(mc), method="temperature").fit(Xw, yw)
R["wine_temperature"] = {"argmax_agree": float((ct.predict(Xw) == mc.predict(Xw)).mean()),
                         "beta": float(ct.calibrated_classifiers_[0].calibrators[0].beta_)}

# 7. ensemble=True averages the five pairs; ensemble=False one pair
ce = CalibratedClassifierCV(GaussianNB(), cv=5, ensemble=True).fit(X_tr, y_tr)
avg = np.mean([c.predict_proba(X_te) for c in ce.calibrated_classifiers_], axis=0)
R["ensemble_mean_diff"] = float(np.abs(avg - ce.predict_proba(X_te)).max())
cf = CalibratedClassifierCV(GaussianNB(), cv=5, ensemble=False).fit(X_tr, y_tr)
R["ens_false_base_is_full_fit"] = float(np.abs(
    cf.calibrated_classifiers_[0].estimator.predict_proba(X_te)
    - GaussianNB().fit(X_tr, y_tr).predict_proba(X_te)).max())


class Count(GaussianNB):
    n = 0

    def fit(self, X, y, sample_weight=None):
        Count.n += 1
        return super().fit(X, y, sample_weight)


for ens in (True, False):
    Count.n = 0
    CalibratedClassifierCV(Count(), cv=5, ensemble=ens).fit(X_tr, y_tr)
    R[f"fits_ensemble_{ens}"] = Count.n
# 8. cv='prefit' and the default estimator
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        CalibratedClassifierCV(nb, cv="prefit").fit(X_ca, y_ca)
        R["prefit"] = "accepted"
except Exception as e:
    R["prefit"] = type(e).__name__ + ": " + str(e)[:200]
# 9. log loss of an exact zero
R["logloss_zero"] = log_loss([1, 0], [0.0, 0.0], labels=[0, 1])
R["neg_log_eps"] = float(-np.log(np.finfo(float).eps))
# 10. Brier for three classes
yb = np.array([0, 1, 2])
pb = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0.0]])
R["brier3_worst"] = brier_score_loss(yb, pb, labels=[0, 1, 2])
R["brier2_worst"] = brier_score_loss([1, 0], [0.0, 1.0])
# 11. forest: distinct probabilities
rf = RandomForestClassifier(random_state=0).fit(X_tr, y_tr)
pr = rf.predict_proba(X_te)[:, 1]
R["rf_multiples_of_0.01"] = bool(np.allclose(pr * 100, np.round(pr * 100)))
R["rf_extreme_share_test"] = float(((pr < 0.02) | (pr > 0.98)).mean())
# 12. the dose data: squared errors of the isotonic fit and of a line
xd = np.arange(1, 9.0)
yd = np.array([2, 5, 3, 4, 1, 6, 5, 7.0])
fd = isotonic_regression(yd)
bl, al = np.polyfit(xd, yd, 1)
R["dose_sse"] = {"isotonic": float(((yd - fd) ** 2).sum()),
                 "line": float(((yd - (al + bl * xd)) ** 2).sum()),
                 "line_a_b": [float(al), float(bl)]}
# 13. AUC of naive Bayes before and after calibration
auc = {"raw": roc_auc_score(y_te, nb.predict_proba(X_te)[:, 1])}
for m in ("sigmoid", "isotonic", "temperature"):
    c = CalibratedClassifierCV(FrozenEstimator(nb), method=m)
    c.fit(X_ca, y_ca)
    pp = c.predict_proba(X_te)[:, 1]
    auc[m] = roc_auc_score(y_te, pp)
    auc[m + "_distinct"] = int(len(np.unique(pp)))
auc["distinct_raw"] = int(len(np.unique(nb.predict_proba(X_te)[:, 1])))
R["auc_nb"] = auc
# 14. breast cancer: exact zeros and ones of the forest
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
Xb, yb = load_breast_cancer(return_X_y=True)
Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(
    Xb, yb, test_size=0.25, random_state=0, stratify=yb)
rfb = RandomForestClassifier(random_state=0).fit(Xb_tr, yb_tr)
pb = rfb.predict_proba(Xb_te)[:, 1]
wrong = ((pb == 0) & (yb_te == 1)) | ((pb == 1) & (yb_te == 0))
R["rf_cancer"] = {"exact_0_or_1": int(((pb == 0) | (pb == 1)).sum()),
                  "confidently_wrong": int(wrong.sum()),
                  "ln_1e-12": float(np.log(1e-12)),
                  "logloss": float(log_loss(yb_te, pb))}
nbb = GaussianNB().fit(Xb_tr, yb_tr)
pn = nbb.predict_proba(Xb_te)[:, 1]
wn = ((pn < 1e-6) & (yb_te == 1)) | ((pn > 1 - 1e-6) & (yb_te == 0))
R["nb_cancer"] = {"wrong_by_more_than_1-1e-6": int(wn.sum()),
                  "logloss": float(log_loss(yb_te, pn))}
json.dump(R, open("../verify/prose_numbers.json", "w"), indent=1)
print(json.dumps(R, indent=1))
