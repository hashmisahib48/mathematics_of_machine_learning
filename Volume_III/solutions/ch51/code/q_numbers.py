# Numbers quoted in the model answers to the questions of Chapter 51 (and Problem 51C.2's companion checks)
import os, sys, json, warnings
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + "/../../../ch51/code")
from scipy.special import expit
from scipy.optimize import brentq
from sklearn.exceptions import ConvergenceWarning
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import make_pipeline
from holes import split, punch, HOLE
warnings.simplefilter("ignore", ConvergenceWarning)

print("=== Question 1: high earners leave the income blank")
rng = np.random.RandomState(0); n = 20000
lin = rng.normal(10.5, 0.6, n); inc = np.exp(lin)               # income, lognormal
z = (inc - inc.mean()) / inc.std()
a = brentq(lambda a: expit(a + 1.5 * z).mean() - 0.2, -30, 30)
blank = rng.uniform(size=n) < expit(a + 1.5 * z)                # P(blank) rises with income (MNAR)
seen = inc[~blank]
print("blank share %.3f   true mean %.0f   mean of seen %.0f (%.1f%% low)" % (blank.mean(), inc.mean(), seen.mean(), 100 * (1 - seen.mean() / inc.mean())))
print("mean of the blank incomes %.0f   mean-filled column: mean %.0f  sd %.0f  (true sd %.0f, ratio of variances %.3f)"
      % (inc[blank].mean(), seen.mean(), np.where(blank, seen.mean(), inc).std(), inc.std(), np.where(blank, seen.mean(), inc).var() / inc.var()))
blank2 = rng.uniform(size=n) < 0.2                              # same rate, MCAR
print("MCAR for comparison: mean of seen %.0f" % inc[~blank2].mean())
top = inc > np.quantile(inc, 0.99)
print("share of the top 1%% of incomes that are blank: MNAR %.2f   MCAR %.2f" % (blank[top].mean(), blank2[top].mean()))

print("\n=== Question 2: small errors, spoiled analysis (Eq. 51.3-1)")
Xtr, Xte, ytr, yte = split(); j = HOLE[0]                       # bmi
res = {k: [] for k in ("complete", "mean", "knn", "iterative", "posterior")}; vr = {k: [] for k in res}; rr = {k: [] for k in res}; rm = []
for s in range(12):
    A = punch(Xtr, "MCAR", 0.2, s); M = np.isnan(A)
    F = {"complete": Xtr, "mean": SimpleImputer().fit_transform(A), "knn": KNNImputer().fit_transform(A),
         "iterative": IterativeImputer(random_state=0).fit_transform(A), "posterior": IterativeImputer(sample_posterior=True, random_state=0).fit_transform(A)}
    for k, Fk in F.items():
        res[k].append(np.corrcoef(Fk[:, j], ytr)[0, 1] ** 2)       # R2 of y on bmi alone
        vr[k].append(Fk[:, j].var() / Xtr[:, j].var())
        rr[k].append(np.corrcoef(Fk[:, j], Xtr[:, 8])[0, 1] / np.corrcoef(Xtr[:, j], Xtr[:, 8])[0, 1])   # corr with s5 (complete)
    rm.append(np.sqrt(np.mean((F["mean"][M[:, j], j] - Xtr[M[:, j], j]) ** 2)) / Xtr[:, j].std())
print("12 patterns of 20%% MCAR blanks in the training table (331 patients), bmi column; mean fill RMSE on its blanks %.2f sd" % np.mean(rm))
print("%-10s %12s %16s %20s" % ("fill", "var ratio", "corr ratio (s5)", "R2 of y on bmi alone"))
for k in res: print("%-10s %12.3f %16.3f %20.3f" % (k, np.mean(vr[k]), np.mean(rr[k]), np.mean(res[k])))
print("theory: variance 1-q = %.2f, correlation sqrt(1-q) = %.3f, with q = 0.2 (actual blank share of bmi %.3f)" % (0.8, np.sqrt(0.8), np.isnan(punch(Xtr, "MCAR", 0.2, 0)[:, j]).mean()))

print("\n=== Question 3: imputing before the folds are cut")
kf = KFold(5, shuffle=True, random_state=0)
for name, imp in (("iterative", IterativeImputer(random_state=0)), ("KNN k=5", KNNImputer()), ("mean", SimpleImputer())):
    diffs = []; inside = []; outside = []
    for s in range(12):
        A = punch(Xtr, "MCAR", 0.3, s)
        i_ = cross_val_score(make_pipeline(imp, LinearRegression()), A, ytr, cv=kf).mean()
        o_ = cross_val_score(LinearRegression(), imp.fit_transform(A), ytr, cv=kf).mean()
        inside.append(i_); outside.append(o_); diffs.append(o_ - i_)
    print("%-10s CV R2 inside the pipeline %.4f   imputed before the folds %.4f   difference %+.4f +- %.4f (12 patterns, 30%% blanks)"
          % (name, np.mean(inside), np.mean(outside), np.mean(diffs), np.std(diffs, ddof=1) / np.sqrt(12)))
# the effect on a single new patient: fill a test row with and without the test rows in the fit
A = punch(Xtr, "MCAR", 0.3, 0); B = punch(Xte, "MCAR", 0.3, 100)
imp_tr = SimpleImputer().fit(A); imp_all = SimpleImputer().fit(np.vstack([A, B]))
print("column means used (bmi): training only %.5f   training+test %.5f" % (imp_tr.statistics_[HOLE[0]], imp_all.statistics_[HOLE[0]]))

print("\n=== Question 4: native NaN against an imputer (book's case study, case.json)")
R = json.load(open(HERE + "/../../../ch51/verify/case.json"))
for kind in ("MCAR", "MAR", "MNAR"):
    d = R[kind + "_0.2"]
    print("%-5s 20%% blanks: native NaN %.3f   mean fill %.3f   iterative %.3f   mean+indicator %.3f" % (kind, np.mean(d["native NaN"]), np.mean(d["mean"]), np.mean(d["iterative"]), np.mean(d["mean+ind"])))
# a blank that carries the answer: a trap for a mean fill, a gift for a tree
rng = np.random.RandomState(1); n = 4000
x = rng.normal(size=n); u = rng.normal(size=n)
ymod = np.where(x > 1, 3.0, 0.0) + 0.5 * u                       # the response jumps when x is large ...
xb = np.where(x > 1, np.nan, x)                                  # ... and exactly then x is blank
tr, te = np.arange(n) < 3000, np.arange(n) >= 3000
for name, m in (("mean fill + linear", make_pipeline(SimpleImputer(), LinearRegression())),
                ("mean fill + indicator + linear", make_pipeline(SimpleImputer(add_indicator=True), LinearRegression())),
                ("HistGradientBoosting, native NaN", HGB(random_state=0))):
    m.fit(xb[tr, None], ymod[tr]); print("%-34s test R2 %.3f" % (name, m.score(xb[te, None], ymod[te])))
PN = json.load(open(HERE + "/../../../ch51/verify/prose_numbers.json")); cl, ch_ = PN["complete_LR"], PN["complete_HGB"]
print("\nloss from the holes against the same model on the complete table (complete: linear %.3f, booster %.3f)" % (cl, ch_))
for kind in ("MCAR", "MAR", "MNAR"):
    d = R[kind + "_0.2"]
    print("%-5s linear + mean fill loses %.3f   native-NaN booster loses %.3f" % (kind, cl - np.mean(d["mean"]), ch_ - np.mean(d["native NaN"])))
