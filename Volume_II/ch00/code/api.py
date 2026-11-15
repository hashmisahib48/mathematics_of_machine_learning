# The estimator pattern: fit, predict, score (the delivery data).
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X = np.array([[1], [2], [3], [4], [5]])      # n = 5 samples, p = 1
y = np.array([3, 4, 6, 7, 10])               # shape (n,)
model = LinearRegression()
model.fit(X, y)                              # learn w and b
print("shapes:", X.shape, y.shape)
print("coef_:", model.coef_.round(4), " intercept_:",
      round(model.intercept_, 4))
print("predict 6 km:", model.predict([[6]]).round(4))
print("score (R^2):", round(model.score(X, y), 4))

pipe = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
pipe.fit(X, y)                               # scale, then fit
print("pipeline predict 6 km:", pipe.predict([[6]]).round(4))
print("step names:", list(pipe.named_steps))
