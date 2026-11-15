# Question 2 of Chapter 55: what repetition fixes
import sys, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../ch55/code')
from cvdata import make_patients, make_sales
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import (KFold, GroupKFold, TimeSeriesSplit, RepeatedKFold, cross_val_score)

# ---- Question 2: what repetition fixes
print("\nQ2  (logistic regression, 60 rows from one population, 50 data sets)")
Xp, yp = make_classification(20000, n_features=40, n_informative=8, n_redundant=4, class_sep=0.8, flip_y=0.05, random_state=0)
Xt, yt = Xp[10000:], yp[10000:]; lr = LogisticRegression(C=1.0, max_iter=500); rng = np.random.RandomState(1)
one, rep, truth, part = [], [], [], []
for d in range(50):
    idx = rng.choice(10000, 60, replace=False); X, y = Xp[idx], yp[idx]
    truth.append(lr.fit(X, y).score(Xt, yt))
    one.append(cross_val_score(lr, X, y, cv=KFold(5, shuffle=True, random_state=0)).mean())
    rep.append(cross_val_score(lr, X, y, cv=RepeatedKFold(n_splits=5, n_repeats=10, random_state=0)).mean())
    if d < 10: part.append(np.std([cross_val_score(lr, X, y, cv=KFold(5, shuffle=True, random_state=r)).mean() for r in range(15)], ddof=1))
one, rep, truth = map(np.array, (one, rep, truth))
print("  sd of the estimate over data sets: one 5-fold %.4f   10 x 5-fold %.4f" % (one.std(ddof=1), rep.std(ddof=1)))
print("  sd over partitions of one data set (5-fold): %.4f" % np.mean(part))
print("  sd of the true accuracy of the fitted model over data sets: %.4f" % truth.std(ddof=1))
print("  rms error against the true accuracy: one %.4f   repeated %.4f" % (np.sqrt(np.mean((one - truth)**2)), np.sqrt(np.mean((rep - truth)**2))))

