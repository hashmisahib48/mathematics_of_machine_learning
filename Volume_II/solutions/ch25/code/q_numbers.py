# Numbers quoted in the model answers to the questions for discussion of Chapter 25.
import numpy as np, warnings
from sklearn.datasets import make_friedman1, load_breast_cancer
from sklearn.ensemble import RandomForestRegressor as RFR, RandomForestClassifier as RFC
from sklearn.model_selection import train_test_split, cross_val_score, GroupKFold, KFold
from sklearn.tree import DecisionTreeRegressor as DTR
np.set_printoptions(precision=4, suppress=True)
warnings.filterwarnings("ignore", message="Some inputs do not have OOB")   # B = 5 and B = 10 on purpose

print("== Q1  which term does each remedy act on? (make_friedman1, 30 training sets of 200 samples)")
Xt, _ = make_friedman1(300, noise=0, random_state=99); _, ft = make_friedman1(300, noise=0, random_state=99)   # noise-free truth at 300 points
cfg = {"single tree": None, "B=100, m=10 (bagging)": dict(n_estimators=100), "B=200, m=10": dict(n_estimators=200),
       "B=100, m=5": dict(n_estimators=100, max_features=0.5), "B=100, max_samples=0.5": dict(n_estimators=100, max_samples=0.5)}
P = {k: [] for k in cfg}
for s in range(30):
    X, y = make_friedman1(200, noise=1.0, random_state=s)
    for k, a in cfg.items():
        m = DTR(random_state=s) if a is None else RFR(random_state=s, **a)
        P[k].append(m.fit(X, y).predict(Xt))
print("%-24s %9s %9s %9s" % ("model", "bias^2", "variance", "sum"))
for k in cfg:
    Q = np.array(P[k]); b2 = ((Q.mean(0) - ft) ** 2).mean(); v = Q.var(0).mean(); print("%-24s %9.3f %9.3f %9.3f" % (k, b2, v, b2 + v))

print("\n== Q2  how many trees vote, how much data does each see (breast cancer)")
d = load_breast_cancer(); Xtr, Xte, ytr, yte = train_test_split(d.data, d.target, test_size=0.25, random_state=0, stratify=d.target)
n = len(ytr); print("distinct samples per tree in a forest on n = %d: %.1f; in a tree of a 5-fold forest (0.8 n): %.1f" % (n, n * (1 - np.exp(-1)), 0.8 * n * (1 - np.exp(-1))))
print("trees voting per sample: OOB 0.368 B;  5-fold CV: B;  final model on the test set: B")
print("   B   OOB error  5-fold error  test error  test error of 0.368 B trees")
for B in (10, 30, 100, 300):
    o, c, t, t37 = [], [], [], []
    for s in range(3):
        f = RFC(B, oob_score=True, random_state=s).fit(Xtr, ytr)
        o.append(1 - f.oob_score_); t.append(1 - f.score(Xte, yte))
        c.append(1 - cross_val_score(RFC(B, random_state=s), Xtr, ytr, cv=KFold(5, shuffle=True, random_state=s)).mean())
        Pr = np.mean([e.predict_proba(Xte) for e in f.estimators_[:int(round(0.368 * B))]], axis=0); t37.append((Pr.argmax(1) != yte).mean())
    print("%4d  %9.4f  %12.4f  %10.4f  %10.4f" % (B, np.mean(o), np.mean(c), np.mean(t), np.mean(t37)))
f5 = RFC(5, oob_score=True, random_state=0).fit(Xtr, ytr); z = (f5.oob_decision_function_.sum(1) == 0).sum()
print("B = 5: samples with no OOB tree: %d of %d (expected %.1f)" % (z, n, n * (1 - np.exp(-1)) ** 5))

print("\n== Q3  what each source of randomness does (breast cancer, 100 trees)")
print("%-34s %12s %12s %10s" % ("configuration", "mean tree acc", "forest acc", "agreement"))
for name, kw in (("no bootstrap, all features", dict(bootstrap=False, max_features=None)), ("bootstrap, all features (bagging)", dict(bootstrap=True, max_features=None)),
                 ("no bootstrap, sqrt features", dict(bootstrap=False, max_features="sqrt")), ("bootstrap, sqrt features (forest)", dict(bootstrap=True, max_features="sqrt"))):
    f = RFC(100, random_state=0, **kw).fit(Xtr, ytr); T = np.array([e.predict(Xte) for e in f.estimators_]); ta = (T == yte).mean(1)
    agree = ((T[:, None, :] == T[None, :, :]).mean(2)[np.triu_indices(100, 1)]).mean()
    print("%-34s %12.4f %12.4f %10.4f" % (name, ta.mean(), f.score(Xte, yte), agree))
print("single tree, random_state=0: test accuracy %.4f" % RFC(1, bootstrap=False, max_features=None, random_state=0).fit(Xtr, ytr).score(Xte, yte))

print("\n== Q4  rows of the same patient (simulation)")
rng = np.random.default_rng(0); P_, R_ = 200, 5
u = rng.normal(size=(P_, 6)); lab = rng.integers(0, 2, P_)             # the label is a coin flip per patient: nothing can be learned
Xg = np.repeat(u, R_, 0) + 0.3 * rng.normal(size=(P_ * R_, 6)); yg = np.repeat(lab, R_); grp = np.repeat(np.arange(P_), R_)
f = RFC(300, oob_score=True, random_state=0).fit(Xg, yg); print("%d patients x %d rows, labels random: OOB accuracy %.4f" % (P_, R_, f.oob_score_))
gk = cross_val_score(RFC(300, random_state=0), Xg, yg, cv=GroupKFold(5), groups=grp).mean(); print("GroupKFold(5) accuracy %.4f (the truth is 0.5)" % gk)
out_in = []                                    # for every tree: is a row's patient in the bag through another row?
for idx in f.estimators_samples_[:100]:
    cnt = np.bincount(grp[idx], minlength=P_)   # draws per patient in this tree
    oob = ~np.isin(np.arange(len(yg)), idx)
    out_in.append((cnt[grp[oob]] > 0).mean())
print("share of OOB (row, tree) pairs whose patient is in the bag by another row: %.4f  (theory 1 - 0.368^4 = %.4f)" % (np.mean(out_in), 1 - np.exp(-1) ** 4))
