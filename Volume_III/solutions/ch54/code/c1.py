# Problem 54C.1: a frozen scaler (and a frozen selector) in cross-validation
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "1"
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = load_breast_cancer(return_X_y=True); cv = StratifiedKFold(5, shuffle=True, random_state=0)
frozen = make_pipeline(FrozenEstimator(StandardScaler().fit(X)), SVC()); normal = make_pipeline(StandardScaler(), SVC())
sf = cross_val_score(frozen, X, y, cv=cv); sn = cross_val_score(normal, X, y, cv=cv)
print("breast cancer, SVC: frozen scaler %.4f, refitted per fold %.4f" % (sf.mean(), sn.mean()))
print("per fold, frozen   :", sf.round(4).tolist()); print("per fold, refitted :", sn.round(4).tolist())
print("largest per-fold difference: %.4f (one row of a fold of %d is %.4f)" % (np.abs(sf - sn).max(), len(y) // 5, 1 / (len(y) // 5)))
sh, sd = [], []
for tr, te in cv.split(X, y):
    sh.append(np.abs(X.mean(0) - X[tr].mean(0)) / X.std(0)); sd.append(np.abs(X.std(0) / X[tr].std(0) - 1))
print("scaler numbers fitted on all rows against the training part of a fold: mean shift %.3f sd, largest %.3f sd; sd differs by %.1f per cent on average" % (np.mean(sh), np.max(sh), 100 * np.mean(sd)))
print("the test part of a fold is %.0f per cent of the rows" % (100 * len(cv.split(X, y).__next__()[1]) / len(y)))

print("\nthe same trick with a selector that reads the labels (60 rows, 2000 noise columns, 30 data sets)")
fr, ho = [], []
for seed in range(30):
    rng = np.random.RandomState(seed); Xn = rng.normal(size=(60, 2000)); yn = rng.randint(0, 2, 60); c5 = StratifiedKFold(5, shuffle=True, random_state=seed)
    sel = SelectKBest(f_classif, k=10).fit(Xn, yn)
    fr.append(cross_val_score(make_pipeline(FrozenEstimator(sel), LogisticRegression(max_iter=1000)), Xn, yn, cv=c5).mean())
    ho.append(cross_val_score(make_pipeline(SelectKBest(f_classif, k=10), LogisticRegression(max_iter=1000)), Xn, yn, cv=c5).mean())
print("frozen selector %.3f +- %.3f   selector refitted per fold %.3f +- %.3f   (chance 0.5)" % (np.mean(fr), np.std(fr), np.mean(ho), np.std(ho)))
