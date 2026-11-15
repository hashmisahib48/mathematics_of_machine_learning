# Problem 1B.2: R^2 on the training half and on the test half
import numpy as np
from sklearn.datasets import make_regression
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X, y = make_regression(n_samples=40, n_features=10, n_informative=3,
                       noise=40.0, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5,
                                          random_state=0)
print("samples per half:", len(y_tr), len(y_te))
print("knobs of the line:", X.shape[1] + 1)
for model in (DummyRegressor(), LinearRegression()):
    model.fit(X_tr, y_tr)                   # learn on training half
    r2_tr = model.score(X_tr, y_tr)               # R^2, training half
    r2_te = model.score(X_te, y_te)               # R^2, test half
    print(f"{type(model).__name__:16s} train {r2_tr:7.3f}"
          f"  test {r2_te:7.3f}")

# Why is the constant model negative on the test half?
gap = (y_tr.mean() - y_te.mean()) ** 2     # squared gap of means
print("means of y:", round(y_tr.mean(), 2), round(y_te.mean(), 2))
print("-gap / var(y_te) =", round(-gap / y_te.var(), 3))
