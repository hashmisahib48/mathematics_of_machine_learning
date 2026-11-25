# Our LinearRegression on the five deliveries of Example 3.4-1.
from linreg import LinearRegression

X = [[2, 1], [4, 2], [5, 1], [7, 3], [9, 2]]      # distance (km), stops
y = [14, 23, 24, 35, 38]                           # delivery time (min)

model = LinearRegression()
model.fit(X, y)
print("coef_          :", [round(w, 4) for w in model.coef_])
print("intercept_     :", round(model.intercept_, 4))
print("n_features_in_ :", model.n_features_in_)
print("predict [6, 2] :", round(model.predict([[6, 2]])[0], 3))
print("score (R^2)    :", round(model.score(X, y), 4))
