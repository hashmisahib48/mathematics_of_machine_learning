# Stacking from scratch, two classes.
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict

AUTO = ("predict_proba", "decision_function", "predict")


def column(out):
    """Level-0 output as one column (two classes)."""
    return out[:, 1:] if out.ndim == 2 else out[:, None]


class MyStack:
    def __init__(self, models, final=None, passthrough=False):
        self.models, self.final = models, final
        self.passthrough = passthrough

    def fit(self, X, y):
        self.methods_ = [next(n for n in AUTO if hasattr(m, n))
                         for m in self.models]
        folds = StratifiedKFold(5)              # what cv=None means
        Z = np.hstack([column(cross_val_predict(    # Eq. 29.4-1
            clone(m), X, y, cv=folds, method=n))
            for m, n in zip(self.models, self.methods_)])
        self.models_ = [clone(m).fit(X, y) for m in self.models]
        self.final_ = clone(self.final or LogisticRegression())
        self.final_.fit(self._meta(Z, X), y)    # Eq. 29.4-2
        return self

    def _meta(self, Z, X):
        return np.hstack([Z, X]) if self.passthrough else Z

    def transform(self, X):                     # Eq. 29.4-3
        return np.hstack([column(getattr(m, n)(X)) for m, n in
                          zip(self.models_, self.methods_)])

    def predict_proba(self, X):
        Z = self.transform(X)
        return self.final_.predict_proba(self._meta(Z, X))

    def predict(self, X):
        return self.final_.classes_[self.predict_proba(X).argmax(1)]
