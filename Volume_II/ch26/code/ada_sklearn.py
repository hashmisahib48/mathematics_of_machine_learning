# AdaBoostClassifier on the six apples, and what it stores.
import numpy as np
from sklearn.ensemble import AdaBoostClassifier

x = np.array([[1], [2], [3], [4], [5], [6.]])
y = np.array([-1, -1, -1, 1, -1, 1])

ada = AdaBoostClassifier(n_estimators=3, random_state=0).fit(x, y)
print("estimator_weights_", ada.estimator_weights_.round(4))
print("estimator_errors_ ", ada.estimator_errors_.round(4))
f = ada.decision_function(x)
a = ada.estimator_weights_
s = sum(w * h.predict(x) for w, h in zip(a, ada.estimators_))
print("decision_function", f.round(4))
print("2 sum(a h)/sum(a)", (2 * s / a.sum()).round(4))
print("predict_proba[:,1]", ada.predict_proba(x)[:, 1].round(4))
print("1/(1+exp(-f))    ", (1 / (1 + np.exp(-f))).round(4))
for k, p in enumerate(ada.staged_predict(x), 1):
    print("after", k, "stumps", p)
sc = ada.staged_score(x, y)
print("staged scores", [round(v, 4) for v in sc])
