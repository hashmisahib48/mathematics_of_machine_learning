# Perceptron is SGDClassifier with fixed settings (Section 17.10).
import numpy as np
from sklearn.linear_model import Perceptron, SGDClassifier

X = np.array([[4, 3], [1, 0], [3, 4], [0, 1]])
y = np.array([1, -1, 1, -1])

per = Perceptron(shuffle=False).fit(X, y)
sgd = SGDClassifier(loss="perceptron", learning_rate="constant",
                    eta0=1.0, penalty=None, shuffle=False,
                    random_state=0).fit(X, y)
for name, m in (("Perceptron", per), ("SGDClassifier", sgd)):
    print(f"{name:13s} w = {m.coef_[0]}  b = {m.intercept_}  "
          f"n_iter_ = {m.n_iter_}  t_ = {m.t_:.0f}")
print("identical:", np.array_equal(per.coef_, sgd.coef_))

small = Perceptron(eta0=0.1, shuffle=False).fit(X, y)
print("eta0 = 0.1:   w =", small.coef_[0].round(4),
      " b =", small.intercept_.round(4))
same = np.array_equal(small.predict(X), per.predict(X))
print("same decisions:", same)
print("score on the four tiles:", per.score(X, y))
