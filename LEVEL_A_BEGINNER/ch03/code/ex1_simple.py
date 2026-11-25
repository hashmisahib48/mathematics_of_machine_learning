# One feature: a straight line
import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])   # hours studied
y = np.array([52.0, 57.0, 61.0, 68.0, 72.0])        # test score

model = LinearRegression()
model.fit(X, y)

print("coef_      :", model.coef_)
print("intercept_ :", model.intercept_)
print("predict 6 h:", model.predict([[6.0]]))
print("R^2 score  :", model.score(X, y))
