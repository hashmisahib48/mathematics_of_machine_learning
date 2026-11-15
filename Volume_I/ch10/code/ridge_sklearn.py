# The two-reading toy solved with scikit-learn.
import textwrap
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge

X = np.array([[1, 1], [2, 2], [3, 3], [4, 5], [5, 4]], dtype=float)
y = np.array([3.0, 4.0, 6.0, 7.0, 10.0])

ols = LinearRegression().fit(X, y)
model = Ridge(alpha=1.0).fit(X, y)         # alpha of Eq. 10.3-1

print("OLS   coef_      =", np.round(ols.coef_, 4))
print("Ridge coef_      =", np.round(model.coef_, 4))
print("Ridge intercept_ =", round(float(model.intercept_), 4))
print("solver_          =", model.solver_)
print("n_iter_          =", model.n_iter_)
print("predict(6, 6)    =", np.round(model.predict([[6, 6]]), 3))
print("score (R2)       =", round(model.score(X, y), 4))

def J(m):                         # Eq. 10.3-1 with alpha = 1
    r = y - m.predict(X)
    return r @ r + 1.0 * m.coef_ @ m.coef_

print("J at ridge solution =", round(J(model), 4))
print("J at OLS weights    =", round(J(ols), 4))
b = y.mean() - X.mean(axis=0) @ model.coef_   # Eq. 10.3-2
print("intercept check     =", round(b, 4))

try:                                       # a solver that needs
    Ridge(solver="lbfgs").fit(X, y)        # positive=True
except ValueError as err:
    print(textwrap.fill("ValueError: " + str(err), 62))
