# The two worked examples of Section 4.4 with our classes.
from linreg import LinearRegression
from ridge import Ridge, poly_features

x = [[-2], [-1], [0], [1], [2]]
y = [4.1, 1.2, 0.1, 0.9, 3.9]
Xq = poly_features(x, 2)                 # columns x and x^2
q = LinearRegression().fit(Xq, y)
print("quadratic: w =", [round(v, 4) for v in q.coef_],
      " b =", round(q.intercept_, 4))
X = [[2, 1], [4, 2], [5, 1], [7, 3], [9, 2]]     # deliveries
t = [14, 23, 24, 35, 38]
for a in (0.0, 1.0, 10.0, 100.0):
    m = Ridge(alpha=a).fit(X, t)
    w = [round(v, 4) for v in m.coef_]
    print(f"alpha = {a:5.1f}: w = {w}  b = {m.intercept_:.3f}")
