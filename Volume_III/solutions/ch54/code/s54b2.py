# Problem 54B.2: how often is the first step fitted?
import tempfile
import numpy as np
from joblib import Memory
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline

class Counting(BaseEstimator, TransformerMixin):
    calls = 0                              # shared by all copies
    def fit(self, X, y=None):
        type(self).calls += 1
        self.mean_ = X.mean(axis=0)
        return self
    def transform(self, X):
        return X - self.mean_

X, y = load_breast_cancer(return_X_y=True)
alphas = (0.1, 1.0, 10.0, 100.0)
cache = Memory(tempfile.mkdtemp(), verbose=0)
for label, memory in (("no memory", None), ("memory", cache)):
    Counting.calls, per_alpha = 0, []
    for a in alphas:                   # one candidate each
        before = Counting.calls
        pipe = Pipeline([("c", Counting()), ("r", Ridge(alpha=a))],
                        memory=memory)
        cross_val_score(pipe, X, y, cv=KFold(5))
        per_alpha.append(Counting.calls - before)
    print(f"{label:10s} per alpha {per_alpha} total {Counting.calls}")
print("4 alphas x 5 folds =", len(alphas) * 5)
