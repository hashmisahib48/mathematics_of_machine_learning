# Problem 16B.1: odds ratio and the hours that give p = 0.8.
import numpy as np
from sklearn.linear_model import LogisticRegression

hours = np.arange(1.0, 9.0)                    # x = 1, ..., 8
passed = np.array([0, 0, 0, 1, 0, 1, 1, 1.0])  # y: 1 = passed
model = LogisticRegression(C=np.inf, tol=1e-10, max_iter=1000)
model.fit(hours[:, None], passed)
b, w = model.intercept_[0], model.coef_[0, 0]
print(f"b = {b:.4f}   w = {w:.4f}   classes_ = {model.classes_}")
print("odds ratio per hour e^w =", round(float(np.exp(w)), 3))

# p = 0.8 means odds 4, so z = b + w x = ln 4 (Eq. 16.2-2)
x80 = (np.log(4) - b) / w
print(f"ln 4 = {np.log(4):.4f}   x = (ln 4 - b)/w = {x80:.3f} hours")
p = model.predict_proba([[x80]])[0]
print("predict_proba at that x:", np.round(p, 4))
print("decision_function there :",
      np.round(model.decision_function([[x80]]), 4))
p56 = model.predict_proba([[5.0], [6.0]])[:, 1]
print("boundary -b/w =", round(float(-b / w), 3),
      " p after 5, 6 hours:", np.round(p56, 4))
