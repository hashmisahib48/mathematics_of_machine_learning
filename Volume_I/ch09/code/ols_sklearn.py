# The delivery problem solved with scikit-learn.
import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])   # shape (5, 1)
y = np.array([3.0, 4.0, 6.0, 7.0, 10.0])            # shape (5,)

model = LinearRegression()      # choose the estimator
model.fit(X, y)                 # learn w and b from the data

print("coef_        =", np.round(model.coef_, 4))
print("intercept_   =", round(float(model.intercept_), 4))
print("rank_        =", model.rank_)
print("singular_    =", np.round(model.singular_, 4))
print("predict(6km) =", np.round(model.predict([[6.0]]), 2))
print("score (R2)   =", round(model.score(X, y), 4))
