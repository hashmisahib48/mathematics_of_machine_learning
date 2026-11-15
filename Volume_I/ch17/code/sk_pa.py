# PassiveAggressiveClassifier is deprecated in scikit-learn 1.9.1.
import textwrap
import warnings
import numpy as np
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import SGDClassifier

X = np.array([[4, 3], [1, 0], [3, 4], [0, 1]])
y = np.array([1, -1, 1, -1])

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    old = PassiveAggressiveClassifier(C=1.0, shuffle=False,
                                      random_state=0).fit(X, y)
print(caught[0].category.__name__)
print(textwrap.fill(str(caught[0].message), 66))

new = SGDClassifier(loss="hinge", penalty=None, shuffle=False,
                    learning_rate="pa1", eta0=1.0,
                    random_state=0).fit(X, y)
print("old:", old.coef_[0].round(4), old.intercept_.round(4))
print("new:", new.coef_[0].round(4), new.intercept_.round(4))

one = SGDClassifier(loss="hinge", penalty=None, learning_rate="pa1",
                    eta0=1.0, shuffle=False)
one.partial_fit(X[:1], y[:1], classes=[-1, 1])   # first tile only
print("after one update:", one.coef_[0], one.intercept_)
print("tau = min(C, loss / |x|^2) = min(1, 1/25) =", 1 / 25)
