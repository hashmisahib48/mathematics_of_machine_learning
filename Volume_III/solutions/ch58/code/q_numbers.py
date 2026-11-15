# Numbers quoted in the model answers to the questions of Chapter 58
import sys, os
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier as GBC, RandomForestClassifier as RF
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, make_scorer
from sklearn.model_selection import StratifiedKFold, TunedThresholdClassifierCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)                    # 0 = malignant
Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
lr = make_pipeline(StandardScaler(), LogisticRegression())

def cost(yt, pred, c_fn, c_fp, pos_label):
    miss = (yt == pos_label) & (pred != pos_label); alarm = (yt != pos_label) & (pred == pos_label)
    return (c_fn * miss.sum() + c_fp * alarm.sum()) / len(yt)

print("== Q1  a forest score of 0.30")
rf = RF(n_estimators=200, random_state=0).fit(Xa, ya); pr = rf.predict_proba(Xb)[:, 0]
pl = lr.fit(Xa, ya).predict_proba(Xb)[:, 0]
edges = np.linspace(0, 1, 6)
for nm, p in (("forest", pr), ("logistic", pl)):
    b = np.digitize(p, edges[1:-1]); print(nm, "(5 equal bins of the score, malignant class)")
    for k in range(5):
        m = b == k
        if m.sum(): print("   bin %.1f-%.1f: n %3d  mean score %.3f  observed share malignant %.3f" % (edges[k], edges[k + 1], m.sum(), p[m].mean(), (yb[m] == 0).mean()))
print("forest scores in [0.2, 0.4):", ((pr >= .2) & (pr < .4)).sum(), " exactly 0.30 +- 0.025:", ((pr > .275) & (pr < .325)).sum(),
      " with a malignant tumour:", ((pr > .275) & (pr < .325) & (yb == 0)).sum())
ts = 1 / 11
print("forest scores >= 1/11:", (pr >= ts).sum(), " logistic scores >= 1/11:", (pl >= ts).sum(), " malignant in test:", (yb == 0).sum())
print("log loss / Brier of P(malignant): forest %.4f / %.4f, logistic %.4f / %.4f"
      % (log_loss(yb == 0, np.clip(pr, 1e-15, 1 - 1e-15)), brier_score_loss(yb == 0, pr), log_loss(yb == 0, pl), brier_score_loss(yb == 0, pl)))
rows = {"forest, predict (0.5)": (rf, 0.5)}
for meth in ("sigmoid", "isotonic"):
    cal = CalibratedClassifierCV(RF(n_estimators=200, random_state=0), method=meth, cv=5).fit(Xa, ya)
    pc = cal.predict_proba(Xb)[:, 0]
    print("%-9s calibrated forest: log loss %.4f, Brier %.4f, scores >= 1/11: %d" % (meth, log_loss(yb == 0, np.clip(pc, 1e-15, 1 - 1e-15)), brier_score_loss(yb == 0, pc), (pc >= ts).sum()))
    rows["forest %s, 1/11" % meth] = pc
def line(nm, p, t):
    pred = np.where(p >= t, 0, 1); hit = ((pred == 0) & (yb == 0)).sum(); miss = ((pred == 1) & (yb == 0)).sum(); fa = ((pred == 0) & (yb == 1)).sum()
    print("%-26s caught %2d missed %d false alarms %2d  cost %.4f" % (nm, hit, miss, fa, cost(yb, pred, 10, 1, 0)))
line("forest raw, 0.5", pr, 0.5); line("forest raw, 1/11", pr, ts); line("forest sigmoid, 1/11", rows["forest sigmoid, 1/11"], ts)
line("forest isotonic, 1/11", rows["forest isotonic, 1/11"], ts); line("logistic, 1/11", pl, ts)
print("cost ratio 10: t* = %.4f" % ts)

print("\n== Q2  best threshold over 20 fold assignments (cost ratio 10)")
c10 = make_scorer(cost, greater_is_better=False, c_fn=10, c_fp=1, pos_label=0)
th, sc, tc = [], [], []
for seed in range(20):
    t = TunedThresholdClassifierCV(lr, scoring=c10, cv=StratifiedKFold(5, shuffle=True, random_state=seed)).fit(Xa, ya)
    th.append(t.best_threshold_); sc.append(-t.best_score_)
    tc.append(cost(yb, t.predict(Xb), 10, 1, 0))
th, sc, tc = map(np.array, (th, sc, tc))
print("best threshold: min %.3f  median %.3f  max %.3f" % (th.min(), np.median(th), th.max()))
print("cross-validated cost at the best threshold: min %.4f  max %.4f" % (sc.min(), sc.max()))
print("test cost at those thresholds: min %.4f  median %.4f  max %.4f   (t* = 1/11: %.4f)"
      % (tc.min(), np.median(tc), tc.max(), cost(yb, np.where(lr.predict_proba(Xb)[:, 0] >= ts, 0, 1), 10, 1, 0)))
t0 = TunedThresholdClassifierCV(lr, scoring=c10, cv=5, store_cv_results=True).fit(Xa, ya)
g, m = t0.cv_results_["thresholds"], -t0.cv_results_["scores"]
ok = g[m <= m.min() + 0.02]
print("seed-0 cost curve: min %.4f at %.3f; thresholds within 0.02 of the minimum run from %.3f to %.3f" % (m.min(), t0.best_threshold_, ok.min(), ok.max()))
print("a patient is one cost unit: 1 miss = 10/426 = %.4f per training patient, 1 false alarm = %.4f" % (10 / 426, 1 / 426))
print("cost of 'always yes' %.4f, 'always no' %.4f per patient (training prevalence %.3f)" % (267 / 426, 10 * 159 / 426, 159 / 426))

print("\n== Q3  permutation importance on training and test rows (signal + 9 noise columns)")
def make(n, seed):
    r = np.random.RandomState(seed); sg = r.standard_normal(n); N = r.standard_normal((n, 9)); yy = (sg + 1.0 * r.standard_normal(n) > 0).astype(int)
    return np.c_[sg, N], yy
Xs, ys = make(200, 0); Xt, yt_ = make(1000, 1)
f = RF(n_estimators=200, random_state=0).fit(Xs, ys)
ptr = permutation_importance(f, Xs, ys, n_repeats=20, random_state=0).importances_mean
pte = permutation_importance(f, Xt, yt_, n_repeats=20, random_state=0).importances_mean
print("accuracy: train %.3f  test %.3f" % (f.score(Xs, ys), f.score(Xt, yt_)))
print("importance of the signal column: train rows %.3f  test rows %.3f" % (ptr[0], pte[0]))
print("sum over the 9 noise columns: train rows %.3f  test rows %.3f ; largest single noise column: train %.3f  test %.3f" % (ptr[1:].sum(), pte[1:].sum(), ptr[1:].max(), pte[1:].max()))
print("impurity share of the noise columns: %.3f" % f.feature_importances_[1:].sum())
print("an honest forest: oob score %.3f" % RF(n_estimators=200, random_state=0, oob_score=True).fit(Xs, ys).oob_score_)
bc = RF(n_estimators=200, random_state=0).fit(Xa, 1 - ya); names = load_breast_cancer().feature_names
pt = permutation_importance(bc, Xa, 1 - ya, n_repeats=10, random_state=0).importances_mean; pv = permutation_importance(bc, Xb, 1 - yb, n_repeats=10, random_state=0).importances_mean
print("breast-cancer forest: accuracy train %.3f test %.3f; sum of the importances train rows %.4f, test rows %.4f" % (bc.score(Xa, 1 - ya), bc.score(Xb, 1 - yb), pt.sum(), pv.sum()))

print("\n== Q4  two reasons for a flat partial dependence")
rg = np.random.RandomState(3); Xi = rg.uniform(-1, 1, (600, 2)); yi = ((Xi[:, 0] * Xi[:, 1] + 0.1 * rg.standard_normal(600)) > 0).astype(int)
gi = GBC(random_state=0).fit(Xi, yi)
rt = np.random.RandomState(4); Xn = rt.uniform(-1, 1, (600, 2)); yn = ((Xn[:, 0] * Xn[:, 1] + 0.1 * rt.standard_normal(600)) > 0).astype(int)
pdi = partial_dependence(gi, Xi, [0], kind="both", grid_resolution=40, method="brute", response_method="predict_proba")
ice, avg = pdi["individual"][0], pdi["average"][0]
rng_ = lambda M_: (M_.max(1) - M_.min(1))          # range of each ICE curve along the grid
print("interaction: PD of x1 from %.3f to %.3f (range %.3f); the range of an ICE curve: mean %.3f, median %.3f; share of curves with range > 0.5: %.3f"
      % (avg.min(), avg.max(), avg.max() - avg.min(), rng_(ice).mean(), np.median(rng_(ice)), (rng_(ice) > 0.5).mean()))
# an irrelevant column: class depends on x1 only; x2 is noise
yo = (Xi[:, 0] + 0.1 * rg.standard_normal(600) > 0).astype(int); go = GBC(random_state=0).fit(Xi, yo)
pdo = partial_dependence(go, Xi, [1], kind="both", grid_resolution=40, method="brute", response_method="predict_proba")
io, ao = pdo["individual"][0], pdo["average"][0]
yon = (Xn[:, 0] + 0.1 * rt.standard_normal(600) > 0).astype(int); po = permutation_importance(go, Xn, yon, n_repeats=20, random_state=0)
print("irrelevant column x2: PD from %.3f to %.3f (range %.3f); the range of an ICE curve: mean %.3f, median %.3f; permutation importance x2 %.4f (x1 %.3f)"
      % (ao.min(), ao.max(), ao.max() - ao.min(), rng_(io).mean(), np.median(rng_(io)), po.importances_mean[1], po.importances_mean[0]))
