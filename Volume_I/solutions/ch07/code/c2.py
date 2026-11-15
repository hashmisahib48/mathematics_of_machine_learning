# Problem 7C.2: what clone, get_params and check_is_fitted do with the broken class
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

class Broken(BaseEstimator, RegressorMixin):          # as in the problem
    def __init__(self, alpha=1.0):
        self.weights = None                          # alpha is not stored
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
    def fit(self, X, y):
        self.w = np.linalg.lstsq(X, y, rcond=None)[0]   # no trailing _
        return self

class Repaired(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=1.0):
        self.alpha = alpha                           # store, nothing else
    def fit(self, X, y):
        if self.alpha < 0:                           # check in fit
            raise ValueError("alpha must be non-negative")
        self.coef_ = np.linalg.lstsq(X, y, rcond=None)[0]
        return self

X = np.array([[1.], [2.], [3.]]); y = np.array([2., 4., 6.])
b = Broken(alpha=0.5)
for what, call in (("get_params()", lambda: b.get_params()),
                   ("clone(b)", lambda: clone(b))):
    try:
        print(what, "->", call())
    except Exception as e:
        print(what, "-> raises", type(e).__name__ + ":", e)
b.fit(X, y)
try:
    check_is_fitted(b); print("check_is_fitted(b): passes")
except NotFittedError:
    print("check_is_fitted(b) after fit: NotFittedError")
try:
    Broken(alpha=-1.0)
except ValueError as e:
    print("Broken(alpha=-1.0) raises at construction:", e)
r = Repaired(alpha=0.5)
print("Repaired:", r.get_params(), clone(r))
r.fit(X, y); check_is_fitted(r); print("fitted: coef_ =", r.coef_)
print("clone of the fitted one has coef_:", hasattr(clone(r), "coef_"))
r.set_params(alpha=-1.0); print("set_params(alpha=-1) accepted:", r.alpha)
try:
    r.fit(X, y)
except ValueError as e:
    print("fit then raises:", e)
