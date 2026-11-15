# Numbers quoted in the model answers to the questions of Chapter 34.
import sys, os, warnings
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE + '/..')
import numpy as np
from sklearn.datasets import make_moons, make_blobs, load_digits
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.semi_supervised import LabelSpreading, LabelPropagation, SelfTrainingClassifier
from sklearn.ensemble import RandomForestClassifier
warnings.simplefilter('ignore', ConvergenceWarning); warnings.simplefilter('ignore', RuntimeWarning); warnings.simplefilter('ignore', UserWarning)
np.set_printoptions(precision=3, suppress=True)

def three(X, y, lab, knn=10, alpha=0.9):
    yy = -np.ones(len(y), int); yy[lab] = y[lab]
    sup = LogisticRegression().fit(X[lab], y[lab]).score(X, y)
    ls = LabelSpreading(kernel='knn', n_neighbors=knn, alpha=alpha, max_iter=5000, tol=1e-6).fit(X, yy).score(X, y)
    st = SelfTrainingClassifier(LogisticRegression(), threshold=0.75, max_iter=None).fit(X, yy).score(X, y)
    return sup, ls, st

print("== Q1  accuracy on all samples: logistic on the labels | LabelSpreading(knn 10, alpha 0.9) | self-training(logistic)")
X, y = make_moons(200, noise=0.1, random_state=1)
i0 = np.where(y == 0)[0][np.argmin(X[y == 0, 0])]; i1 = np.where(y == 1)[0][np.argmax(X[y == 1, 0])]
print("(a) two moons, one label each      :", np.round(three(X, y, np.array([i0, i1])), 3))
rng = np.random.default_rng(0)
res = []
for s in range(100):                                     # (b) overlapping clouds, 6 labels, 100 draws
    Xb, yb = make_blobs(400, centers=[[-1.5, 0], [1.5, 0]], cluster_std=1.6, random_state=1000 + s)
    lab = np.r_[rng.choice(np.where(yb == 0)[0], 3, replace=False), rng.choice(np.where(yb == 1)[0], 3, replace=False)]
    res.append(three(Xb, yb, lab))
res = np.array(res); print("(b) overlapping clouds, mean of 100 draws:", res.mean(0).round(3), " best possible 0.823")
print("    share of draws with spreading below supervised: %.2f, self-training below: %.2f" % ((res[:, 1] < res[:, 0]).mean(), (res[:, 2] < res[:, 0]).mean()))
def spiral(n, noise, seed):                             # two interleaved arms
    r = np.random.default_rng(seed); t = np.sqrt(r.uniform(0.04, 1, n)) * 3.2 * np.pi
    a = np.c_[t * np.cos(t), t * np.sin(t)] / 10; Xs = np.r_[a, -a] + noise * r.standard_normal((2 * n, 2))
    return Xs, np.r_[np.zeros(n, int), np.ones(n, int)]
for n in (150,):
    Xs, ys = spiral(n, 0.02, 3); r = np.random.default_rng(1)
    out = []
    for s in range(20):
        lab = np.r_[r.choice(np.where(ys == 0)[0], 2, replace=False), r.choice(np.where(ys == 1)[0], 2, replace=False)]
        out.append(three(Xs, ys, lab))
    print("(c) two spirals, %d samples per arm, 2 labels each, mean of 20 draws:" % n, np.mean(out, 0).round(3))

print("\n== Q2  ten per cent wrong labels: how many wrong labels does soft clamping repair? (digits, knn 7)")
Xd, yd = load_digits(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(Xd, yd, test_size=0.25, random_state=0, stratify=yd)
print(f"{'labels':>7s} {'model':>20s} {'repaired':>9s} {'right lost':>11s} {'acc unlabelled':>15s}")
for nl in (100, 300, 600):
    lab, _ = train_test_split(np.arange(len(ytr)), train_size=nl, random_state=0, stratify=ytr)
    r = np.random.default_rng(1); bad = r.choice(lab, nl // 10, replace=False)
    yn = ytr.copy(); yn[bad] = (yn[bad] + r.integers(1, 10, len(bad))) % 10
    yy = -np.ones(len(ytr), int); yy[lab] = yn[lab]; U = yy == -1; good = np.setdiff1d(lab, bad)
    for name, m in [('propagation', LabelPropagation(kernel='knn', max_iter=1000))] + [('spreading a=%.2f' % a, LabelSpreading(kernel='knn', alpha=a, max_iter=500)) for a in (0.2, 0.8, 0.99)]:
        t = m.fit(Xtr, yy).transduction_
        print(f"{nl:7d} {name:>20s} {int((t[bad] == ytr[bad]).sum()):4d} of {len(bad):3d} {int((t[good] != ytr[good]).sum()):6d} of {len(good):3d} {np.mean(t[U] == ytr[U]):13.3f}")

print("\n== Q3  Example 34.5-2 with other settings")
x = np.array([0, 1, 2, 3.4, 4.2, 5, 6, 7.]).reshape(-1, 1); y8 = np.array([0, 0, -1, -1, -1, -1, 1, 1])
from sklearn.neighbors import KNeighborsClassifier
for kw in (dict(threshold=0.75), dict(threshold=0.4), dict(criterion='k_best', k_best=1, threshold=0.75)):
    st = SelfTrainingClassifier(KNeighborsClassifier(2), max_iter=10, **kw).fit(x, y8)
    print(kw, "-> n_iter_", st.n_iter_, "termination", st.termination_condition_, "labeled_iter_", st.labeled_iter_.tolist(), "transduction_", st.transduction_.tolist())
print("probabilities of the 2-NN on the first round for x = 2, 3.4, 4.2, 5:", KNeighborsClassifier(2).fit(x[[0, 1, 6, 7]], y8[[0, 1, 6, 7]]).predict_proba(x[[2, 3, 4, 5]])[:, 1])
Xb, yb = make_blobs(400, centers=[[-1.5, 0], [1.5, 0]], cluster_std=1.6, random_state=5)
Xbt, ybt = make_blobs(4000, centers=[[-1.5, 0], [1.5, 0]], cluster_std=1.6, random_state=777)
r = np.random.default_rng(5); idx = np.r_[r.choice(np.where(yb == 0)[0], 5, replace=False), r.choice(np.where(yb == 1)[0], 5, replace=False)]
lr = LogisticRegression().fit(Xb[idx], yb[idx]); rf = RandomForestClassifier(3, random_state=0).fit(Xb[idx], yb[idx])
for name, m in (('logistic', lr), ('forest of 3 trees', rf)):
    p = m.predict_proba(Xb)[:, 1]; conf = np.maximum(p, 1 - p)
    print(name, "distinct confidences on 400 samples:", np.unique(conf.round(3))[:6], "...", len(np.unique(conf.round(3))), "values")
    for t in (0.6, 0.75, 0.9, 0.95):
        sel = conf > t; ok = (m.predict(Xb)[sel] == yb[sel]).mean() if sel.any() else float('nan')
        print("   threshold %.2f: %3d samples selected, %.3f of them right" % (t, sel.sum(), ok))
print("forest of 3 trees, share of samples with confidence exactly 1:", (np.maximum(rf.predict_proba(Xb)[:, 1], 1 - rf.predict_proba(Xb)[:, 1]) == 1).mean().round(3))

print("\n== Q4  cross_val_score with -1 in y")
lab, _ = train_test_split(np.arange(len(ytr)), train_size=30, random_state=0, stratify=ytr)
yy = -np.ones(len(ytr), int); yy[lab] = ytr[lab]
for name, mk in [('LabelSpreading() default', lambda: LabelSpreading()), ('LabelSpreading(knn 7, 0.8)', lambda: LabelSpreading(kernel='knn', alpha=0.8))]:
    print(name, "cross_val_score on 30 labels:", cross_val_score(mk(), Xtr, yy, cv=5).round(3), " share of labelled samples: %.3f" % (30 / len(ytr)))
for nunl in (0.5, 0.1, 0.03):
    r = np.random.default_rng(0); y_ = ytr.copy(); y_[r.random(len(y_)) < nunl] = -1
    print("   share unlabelled %.2f: cross_val_score(knn 7) = %.3f (labelled share %.3f)" % (nunl, cross_val_score(LabelSpreading(kernel='knn', alpha=0.8), Xtr, y_, cv=5).mean(), (y_ != -1).mean()))
res = []
for seed in range(5):
    lab, _ = train_test_split(np.arange(len(ytr)), train_size=30, random_state=seed, stratify=ytr)
    y_ = -np.ones(len(ytr), int); y_[lab] = ytr[lab]; U = y_ == -1
    m = LabelSpreading(kernel='knn', alpha=0.8).fit(Xtr, y_)
    res.append([np.mean(m.transduction_[U] == ytr[U]), m.score(Xte, yte), np.mean(m.predict(Xtr[U]) == ytr[U])])
res = np.array(res); print("transductive (unlabelled train) / inductive (test) / predict on unlabelled train, mean of 5 draws:", res.mean(0).round(3), " sd", res.std(0).round(3))
