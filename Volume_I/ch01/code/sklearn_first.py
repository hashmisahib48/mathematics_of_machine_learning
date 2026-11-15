# The same two steps for every estimator: fit, then predict.
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])   # km
y = np.array([3.0, 4.0, 6.0, 7.0, 10.0])            # minutes
X_new = np.array([[6.0]])       # a delivery that happens later ...
y_new = 12.0                    # ... and takes 12 minutes
models = (DummyRegressor(strategy="mean"), LinearRegression())
for model in models:
    model.fit(X, y)                         # learn the knobs
    y_hat = model.predict(X_new)[0]         # use them
    print(type(model).__name__)
    print("   R2 on the training data :",
          round(model.score(X, y), 4))
    print("   prediction for 6 km     :", round(float(y_hat), 2))
    print("   error on the new delivery:", round(y_new - y_hat, 2))
print("constant_ =", models[0].constant_.ravel())
print("coef_, intercept_ =", models[1].coef_,
      round(float(models[1].intercept_), 2))
