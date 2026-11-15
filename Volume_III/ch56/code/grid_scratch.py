# Section 56.6: a grid search in a few lines, checked against sklearn.
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import (GridSearchCV, ParameterGrid,
                                     StratifiedKFold)
from common import cancer_split, svc_pipe, GRID


class MiniGridSearch:
    def __init__(self, estimator, grid, cv):
        self.estimator, self.grid, self.cv = estimator, grid, cv

    def fit(self, X, y):
        folds = list(self.cv.split(X, y))      # one set of folds
        self.params_ = list(ParameterGrid(self.grid))
        self.scores_ = np.empty((len(self.params_), len(folds)))
        for i, p in enumerate(self.params_):
            for k, (tr, te) in enumerate(folds):
                m = clone(self.estimator).set_params(**p)
                self.scores_[i, k] = m.fit(X[tr], y[tr]).score(
                    X[te], y[te])                       # s_k
        self.mean_ = self.scores_.mean(axis=1)          # Eq. 56.3-1
        self.best_index_ = int(np.argmax(self.mean_))   # first max
        self.best_params_ = self.params_[self.best_index_]
        self.best_estimator_ = clone(self.estimator).set_params(
            **self.best_params_).fit(X, y)              # refit
        return self

Xtr, Xte, ytr, yte = cancer_split()
mine = MiniGridSearch(svc_pipe(), GRID,
                      StratifiedKFold(5)).fit(Xtr, ytr)
ref = GridSearchCV(svc_pipe(), GRID, cv=5).fit(Xtr, ytr)
d = np.abs(mine.mean_ - ref.cv_results_["mean_test_score"]).max()
assert d < 1e-12 and mine.best_index_ == ref.best_index_
assert mine.best_params_ == ref.best_params_
print("largest difference of mean scores:", d)
print("best index", mine.best_index_,
      " CV", round(mine.mean_.max(), 4),
      " test", round(mine.best_estimator_.score(Xte, yte), 4),
      round(ref.score(Xte, yte), 4))
