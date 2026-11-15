# Problem 55B.2: a fixed train/test split written as a PredefinedSplit
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (PredefinedSplit, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
Xa, Xt, ya, yt = train_test_split(X, y, test_size=0.25,
                                  random_state=0, stratify=y)
pipe = make_pipeline(StandardScaler(),
                     LogisticRegression(max_iter=1000))

# 426 training rows, then 143 test rows; -1 = never tested
Xall, yall = np.r_[Xa, Xt], np.r_[ya, yt]
test_fold = np.r_[-np.ones(len(ya), int), np.zeros(len(yt), int)]
cv = PredefinedSplit(test_fold)
print("rows:", len(ya), "+", len(yt), " n_splits:", cv.get_n_splits())

tr, te = next(iter(cv.split()))
print("train rows 0..%d (%d)  test rows %d..%d (%d)"
      % (tr[-1], len(tr), te[0], te[-1], len(te)))

cvs = cross_val_score(pipe, Xall, yall, cv=cv)   # one fold, one score
hold = pipe.fit(Xa, ya).score(Xt, yt)            # the same by hand
print("cross_val_score: %.4f  hold-out: %.4f  equal: %s"
      % (cvs[0], hold, np.isclose(cvs[0], hold)))
print("correct test rows: %d of %d"
      % (round(hold * len(yt)), len(yt)))
