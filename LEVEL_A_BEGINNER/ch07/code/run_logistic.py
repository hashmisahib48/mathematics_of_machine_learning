# The worked example of Section 7.4: hours of study and the result.
from logistic import LogisticRegression
import math

X = [[1], [2], [3], [4], [5], [6]]       # hours of study
y = [0, 0, 1, 0, 1, 1]                   # 1 = passed
m = LogisticRegression(eta=0.1, n_iter=1).fit(X, y)
print("one step:  w =", round(m.coef_[0], 4), " b =", m.intercept_)
m = LogisticRegression(eta=0.5, n_iter=20000, tol=1e-8).fit(X, y)
w, b = m.coef_[0], m.intercept_
print(f"fitted:    w = {w:.4f}  b = {b:.4f}  iterations {m.n_iter_}")
print(f"cost:      {m.cost_[0]:.4f} -> {m.cost_[-1]:.4f}")
print(f"odds ratio e^w = {math.exp(w):.3f}  boundary x = {-b / w:.3f}")
print("P(pass | 5 h) =", round(m.predict_proba([[5]])[0], 4))
print("training accuracy", round(m.score(X, y), 3))
