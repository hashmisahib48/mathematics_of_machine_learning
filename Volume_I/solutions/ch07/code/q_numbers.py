# Numbers quoted in the model answers of Chapter 7
import math
import numpy as np
import warnings
from scipy.special import softmax, expit
from sklearn.base import BaseEstimator, clone
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

print("== Q1  a constructor that does work")
class Doubler(BaseEstimator):
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.alpha2 = 2 * alpha          # derived in the constructor
m = Doubler(alpha=1.0).set_params(alpha=3.0)
print("after set_params(alpha=3): alpha =", m.alpha, " alpha2 =", m.alpha2, " (should be 6)")
c = clone(Doubler(alpha=5.0)); print("clone(alpha=5): alpha =", c.alpha, " alpha2 =", c.alpha2)
class Copier(BaseEstimator):
    def __init__(self, alpha=1.0):
        self.alpha = alpha * 1.0         # builds a new object from the argument
try:
    clone(Copier(alpha=2.5)); print("clone worked")
except RuntimeError as e: print("clone(Copier(2.5)) raises RuntimeError:", str(e)[:60], "...")
class Checker(BaseEstimator):
    def __init__(self, alpha=1.0):
        if alpha < 0: raise ValueError("alpha must be >= 0")
        self.alpha = alpha
ch = Checker(alpha=1.0)
ch.set_params(alpha=-1.0); print("set_params(alpha=-1) on Checker accepted silently; alpha =", ch.alpha)
try: Checker(alpha=-1.0)
except ValueError as e: print("Checker(alpha=-1) raises at construction:", e)
try: LogisticRegression(C=-1.0); print("LogisticRegression(C=-1.0) is accepted by the constructor")
except Exception as e: print("raises", e)

print("\n== Q2  predict_proba or decision_function")
f = np.array([-4.728, 0.643, 4.085]); print("scores of Example 7.5-1:", f, " order:", np.argsort(f).tolist())
print("softmax:", softmax(f).round(4), " order:", np.argsort(softmax(f)).tolist())
for s in (10, 20, 40, 45):
    print(f"binary score {s:2d}: P = expit = {expit(s)!r}  1 - P = {1 - expit(s):.3e}")
print("scores 40 and 45 give equal probabilities:", expit(40) == expit(45))
print("tie in float64 from score:", next(s for s in range(1, 60) if expit(s) == 1.0))
X, y = load_iris(return_X_y=True)
lr = LogisticRegression(max_iter=1000).fit(X, y)
d = lr.decision_function(X); P = lr.predict_proba(X)
print("iris: argmax agree for all 150:", bool((d.argmax(1) == P.argmax(1)).all()))
from sklearn.svm import LinearSVC
print("hasattr LinearSVC predict_proba:", hasattr(LinearSVC(), "predict_proba"), " decision_function:", hasattr(LinearSVC(), "decision_function"))

print("\n== Q3  statistics from the wrong rows")
tr = np.array([2., 4, 6, 8]); mu, sd = tr.mean(), tr.std()
print("training mean, sd:", mu, round(sd, 4), " z(10) with training numbers:", round((10 - mu) / sd, 4))
al = np.append(tr, 10.); print("with 10 included: mean", al.mean(), " sd", round(al.std(), 4), " z(10) =", round((10 - al.mean()) / al.std(), 4))
print("test part {10}: own mean/sd -> sd =", np.array([10.]).std(), " z = 0 whatever the value (scale_ is set to 1):")
from sklearn.preprocessing import StandardScaler
for v in (10., 100.):
    print("   StandardScaler fitted on [[%g]] gives z =" % v, StandardScaler().fit([[v]]).transform([[v]]).ravel())
te = np.array([10., 12.]); print("test part {10, 12}: correct z =", ((te - mu) / sd).round(4), " own statistics z =", ((te - te.mean()) / te.std()).round(4))

print("\n== Q4  one error in 38")
a, m_ = 37 / 38, 38
s = math.sqrt(a * (1 - a) / m_); print("a =", round(a, 4), " s =", round(s, 4), " one flower = 1/38 =", round(1 / 38, 4), " ratio", round((1 / 38) / s, 2))
print("interval a +- s:", round(a - s, 4), round(a + s, 4), "   a +- 2s:", round(a - 2 * s, 4), round(a + 2 * s, 4))
m4 = 4 * m_; s4 = math.sqrt(a * (1 - a) / m4); print("m =", m4, " s =", round(s4, 4), " ratio", round(s / s4, 3))
m_third = a * (1 - a) / 0.0005 ** 2; print("m for s = 0.0005 (third decimal):", round(m_third))
