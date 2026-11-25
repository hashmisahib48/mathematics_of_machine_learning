# Plain Python, NumPy and scikit-learn on the same data.
import numpy as np
import linreg
import linreg_numpy
from sklearn.linear_model import LinearRegression

X = [[2, 1], [4, 2], [5, 1], [7, 3], [9, 2]]
y = [14, 23, 24, 35, 38]

for name, model in [("plain Python", linreg.LinearRegression()),
                    ("NumPy", linreg_numpy.LinearRegression()),
                    ("scikit-learn", LinearRegression())]:
    model.fit(X, y)
    print(f"{name:12s} coef_={np.round(model.coef_, 4)}"
          f" intercept_={model.intercept_:.4f}"
          f" predict={np.round(model.predict([[6, 2]]), 3)}")
