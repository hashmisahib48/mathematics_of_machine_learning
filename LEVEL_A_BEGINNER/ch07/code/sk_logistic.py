# The same example with scikit-learn's LogisticRegression.
import numpy as np
from sklearn.linear_model import LogisticRegression

X = np.array([[1], [2], [3], [4], [5], [6]])
y = np.array([0, 0, 1, 0, 1, 1])
m = LogisticRegression(C=np.inf).fit(X, y)       # no penalty
print("C = inf: w =", m.coef_.round(4), " b =", m.intercept_.round(4))
print("P(fail), P(pass) at 5 h:", m.predict_proba([[5]]).round(4))
print("score t at 5 h:", m.decision_function([[5]]).round(4))
r = LogisticRegression(C=1.0).fit(X, y)          # default penalty
print("C = 1:   w =", r.coef_.round(4), " b =", r.intercept_.round(4))
