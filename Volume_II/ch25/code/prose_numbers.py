# Numbers quoted in the prose of Chapter 25 (not listings).
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)
rf = RandomForestClassifier(n_estimators=100, oob_score=True,
                            random_state=0).fit(X_tr, y_tr)
print("features per split (p = 30):", rf.estimators_[0].max_features_)
acc = np.array([t.score(X_te, y_te) for t in rf.estimators_])
print("single trees of the forest: test accuracy mean "
      f"{acc.mean():.4f}, min {acc.min():.4f}, max {acc.max():.4f}")
print("forest test accuracy:", round(rf.score(X_te, y_te), 4))
print("trees better than the forest:", int((acc > rf.score(X_te, y_te)).sum()))
print("leaves per tree: mean", np.mean([t.get_n_leaves() for t in rf.estimators_]),
      " depth mean", np.mean([t.get_depth() for t in rf.estimators_]))
full = DecisionTreeClassifier(random_state=0).fit(X_tr, y_tr)
print("one full tree: leaves", full.get_n_leaves(), "depth", full.get_depth())

n = len(y_tr)
inbag = np.array([np.bincount(s, minlength=n) for s in rf.estimators_samples_])
n_oob = (inbag == 0).sum(axis=0)
print("trees out of bag per sample: mean", n_oob.mean(),
      "min", n_oob.min(), "max", n_oob.max())

votes = np.mean([t.predict(X_te) for t in rf.estimators_], axis=0) > 0.5
print("hard vote differs from averaged probabilities on",
      int((votes.astype(int) != rf.predict(X_te)).sum()), "test patients")

et = ExtraTreesClassifier(n_estimators=100, random_state=0).fit(X_tr, y_tr)
print("extra trees: leaves per tree mean",
      np.mean([t.get_n_leaves() for t in et.estimators_]),
      "bootstrap", et.bootstrap)
p = rf.predict_proba(X_te)[:, 1]
print("forest P(benign) on test: share of patients with P in (0.1, 0.9):",
      round(float(((p > 0.1) & (p < 0.9)).mean()), 4))

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    few = RandomForestClassifier(n_estimators=5, oob_score=True,
                                 random_state=0).fit(X, y)
zero = few.oob_decision_function_.sum(axis=1) == 0
print("5 trees on all 569 patients: rows of oob_decision_function_ "
      "that are all zero:", int(zero.sum()),
      " expected", round(569 * (1 - np.exp(-1)) ** 5, 1),
      " oob_score_", round(few.oob_score_, 4))

r1 = RandomForestClassifier(n_estimators=50, random_state=0, n_jobs=2).fit(X, y)
r2 = RandomForestClassifier(n_estimators=50, random_state=0).fit(X, y)
print("n_jobs=2 vs None: largest difference in predict_proba:",
      float(np.abs(r1.predict_proba(X) - r2.predict_proba(X)).max()))
