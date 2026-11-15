# PolynomialFeatures: columns, names, counts, and a fit.
import numpy as np
from math import comb
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

X = np.array([[2.0, 3.0], [1.0, -1.0]])      # n = 2, p = 2
poly = PolynomialFeatures(degree=2).fit(X)   # other defaults
print(poly.get_feature_names_out(["a", "b"]))
print(poly.transform(X))
print("n_output_features_ =", poly.n_output_features_)
print("powers_ =", poly.powers_.tolist())
io = PolynomialFeatures(3, interaction_only=True,
                        include_bias=False)
print(io.fit(np.zeros((1, 3))).get_feature_names_out(list("abc")))
for p, d in ((5, 3), (10, 2), (50, 3)):
    pf = PolynomialFeatures(d).fit(np.zeros((1, p)))
    n = pf.n_output_features_
    print(f"p = {p:2d}, d = {d}: {n:5d} columns;",
          f"C(p+d, d) = {comb(p + d, d)}")

x = np.arange(1.0, 6.0)[:, None]             # the battery data
y = np.array([2.0, 3.0, 8.0, 15.0, 22.0])
fit = make_pipeline(PolynomialFeatures(2), LinearRegression())
fit.fit(x, y)
print("weights of 1, x, x^2:", np.round(fit[-1].coef_, 4))
print("intercept:", round(float(fit[-1].intercept_), 4))
