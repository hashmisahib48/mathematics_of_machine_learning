# scikit-learn's PolynomialFeatures and Ridge give the same numbers.
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge

x = [[-2], [-1], [0], [1], [2]]
y = [4.1, 1.2, 0.1, 0.9, 3.9]
pf = PolynomialFeatures(degree=2, include_bias=False)
Xq = pf.fit_transform(x)
print(pf.get_feature_names_out(), Xq[0])
q = LinearRegression().fit(Xq, y)
print("quadratic:", q.coef_.round(4), round(q.intercept_, 4))
X = [[2, 1], [4, 2], [5, 1], [7, 3], [9, 2]]
t = [14, 23, 24, 35, 38]
r = Ridge(alpha=10.0).fit(X, t)
print("ridge, alpha = 10:", r.coef_.round(4), round(r.intercept_, 3))
