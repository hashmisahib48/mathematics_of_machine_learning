# Problem 55B.1: repeated stratified 5-fold CV of a logistic model
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (RepeatedStratifiedKFold,
                                     cross_val_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
pipe = make_pipeline(StandardScaler(),
                     LogisticRegression(max_iter=1000))
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)
s = cross_val_score(pipe, X, y, cv=cv)    # 50 scores
rep = s.reshape(10, 5).mean(axis=1)       # one mean per repeat

print("mean of the 50 scores      : %.4f" % s.mean())
print("sd of the 50 scores        : %.4f" % s.std(ddof=1))
print("sd of the 10 repeat means  : %.4f" % rep.std(ddof=1))
print("repeat means: min %.4f  max %.4f" % (rep.min(), rep.max()))

p, n = s.mean(), len(y)
b1 = np.sqrt(p * (1 - p) / (n / 5))       # binomial sd of one fold
b2 = np.sqrt(p * (1 - p) / n)             # ... of the whole sample
print("\nbinomial sd of one fold (%.1f rows): %.4f" % (n / 5, b1))
print("binomial sd of the whole sample (n = %d): %.4f" % (n, b2))
print("sd(50) / sqrt(50), a false standard error: %.4f"
      % (s.std(ddof=1) / np.sqrt(50)))
