# The pass/fail problem solved with scikit-learn.
import numpy as np
from sklearn.linear_model import LogisticRegression

X = np.arange(1.0, 9.0).reshape(-1, 1)         # hours, shape (8, 1)
y = np.array([0, 0, 0, 1, 0, 1, 1, 1])         # 1 = passed

model = LogisticRegression()                   # C = 1, L2 penalty
model.fit(X, y)

print("classes_      =", model.classes_)
print("coef_         =", np.round(model.coef_, 4))
print("intercept_    =", np.round(model.intercept_, 4))
print("n_iter_       =", model.n_iter_)
new = np.array([[2.0], [4.0], [5.0], [7.0]])
print("predict_proba =\n", np.round(model.predict_proba(new), 4))
z = model.decision_function(new)
print("decision_fn   =", np.round(z, 4))
print("sigmoid(z)    =", np.round(1 / (1 + np.exp(-z)), 4))
print("predict       =", model.predict(new))
print("odds ratio    =", np.round(np.exp(model.coef_), 3))

free = LogisticRegression(C=np.inf).fit(X, y)  # no penalty
print("C = inf       =", np.round(free.coef_, 4),
      np.round(free.intercept_, 4))

cw = LogisticRegression(class_weight={0: 1, 1: 3}).fit(X, y)
sw = LogisticRegression().fit(X, y, sample_weight=3.0 ** y)
print("class_weight vs sample_weight, max |dw| =",
      float(abs(cw.coef_ - sw.coef_).max()))
