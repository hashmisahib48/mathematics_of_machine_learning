# Which losses give probabilities? (COMMON ERROR of Section 17.10)
import numpy as np
from sklearn.linear_model import SGDClassifier

X = np.array([[4, 3], [1, 0], [3, 4], [0, 1]], dtype=float)
y = np.array([1, 0, 1, 0])

hinge = SGDClassifier(loss="hinge", random_state=0).fit(X, y)
try:
    hinge.predict_proba(X)
except AttributeError as err:
    print("AttributeError:", err)

mh = SGDClassifier(loss="modified_huber", alpha=1.0,
                   random_state=0).fit(X, y)
f = mh.decision_function(X)
print("f(x)                   ", f.round(3))
print("predict_proba, class 1", mh.predict_proba(X)[:, 1].round(3))
q = (np.clip(f, -1, 1) + 1) / 2
print("(clip(f, -1, 1) + 1)/2", q.round(3))

lg = SGDClassifier(loss="log_loss", alpha=1.0,
                   random_state=0).fit(X, y)
f = lg.decision_function(X)
p = lg.predict_proba(X)[:, 1]
print("log_loss: 1/(1+exp(-f)) =", (1 / (1 + np.exp(-f))).round(3))
print("log_loss: predict_proba =", p.round(3))
