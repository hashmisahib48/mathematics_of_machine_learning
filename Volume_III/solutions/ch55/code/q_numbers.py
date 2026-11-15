# Numbers quoted in the model answers to the questions for discussion of Chapter 55
import sys, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../ch55/code')
from cvdata import make_patients, make_sales
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import (KFold, GroupKFold, TimeSeriesSplit, RepeatedKFold, cross_val_score)

# ---- Question 1: patients in both training and test rows
print("Q1  (8 invented hospitals, 40 patients x 8 visits, forest of 50 trees)")
rf = RandomForestClassifier(50, random_state=0); sh, gr, tr, share = [], [], [], []
for seed in range(8):
    X, y, g = make_patients(40, 8, seed=seed); Xn, yn, _ = make_patients(150, 8, seed=100 + seed)
    cv = KFold(5, shuffle=True, random_state=0)
    sh.append(cross_val_score(rf, X, y, cv=cv).mean())
    gr.append(cross_val_score(rf, X, y, groups=g, cv=GroupKFold(5)).mean())
    tr.append(rf.fit(X, y).score(Xn, yn))
    share.append(np.mean([np.isin(g[te], g[trn]).mean() for trn, te in cv.split(X)]))
print("  shuffled KFold %.3f   GroupKFold %.3f   new patients %.3f" % (np.mean(sh), np.mean(gr), np.mean(tr)))
print("  share of held-out rows whose patient is also in training, shuffled KFold: %.3f" % np.mean(share))
s_a, s, m = 2.0, 1.0, 8
print("  Eq. 55.3-3: MSE same %.3f  new %.1f  ratio %.2f" % (s**2 * (1 + 1 / (m - 1)), s_a**2 + s**2, (s_a**2 + s**2) / (s**2 * (1 + 1 / (m - 1)))))

# ---- Question 4: the winner's curse (500 repetitions, the setting of §55.8)
print("\nQ4  (labels are coin flips; best of 20 rules)")
def fold_acc(X, y, tr, te):
    m0, m1 = X[tr][y[tr] == 0].mean(0), X[tr][y[tr] == 1].mean(0)
    return ((np.abs(X[te] - m1) < np.abs(X[te] - m0)) == y[te][:, None].astype(bool)).mean(0)
def cv_acc(X, y, rows, K, seed):
    return np.mean([fold_acc(X, y, rows[a], rows[b]) for a, b in KFold(K, shuffle=True, random_state=seed).split(rows)], axis=0)
rng = np.random.RandomState(0); best, nest = [], []
for rep_ in range(500):
    X, y = rng.normal(size=(60, 20)), rng.randint(0, 2, 60); rows = np.arange(60)
    best.append(cv_acc(X, y, rows, 5, rep_).max()); out = []
    for a, b in KFold(5, shuffle=True, random_state=rep_).split(X):
        out.append(fold_acc(X, y, a, b)[cv_acc(X, y, a, 4, rep_).argmax()])
    nest.append(np.mean(out)); rng.normal(size=(5000, 20)); rng.randint(0, 2, 5000)   # keep the stream of the book
print("  best by 5-fold CV: mean %.3f sd %.3f   nested: mean %.3f sd %.3f" % (np.mean(best), np.std(best, ddof=1), np.mean(nest), np.std(nest, ddof=1)))
print("  nested estimates within 0.1 of 0.5: %.2f of the repetitions" % np.mean(np.abs(np.array(nest) - 0.5) < 0.1))
