# The same estimator with NumPy arrays.
import numpy as np


class LinearRegression:
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        p = X.shape[1]
        x_mean = X.mean(axis=0) if self.fit_intercept else np.zeros(p)
        y_mean = y.mean() if self.fit_intercept else 0.0
        Xc, yc = X - x_mean, y - y_mean                   # Step 1
        A, c = Xc.T @ Xc, Xc.T @ yc                       # Steps 2-3
        self.coef_ = np.linalg.solve(A, c)                # Steps 4-8
        self.intercept_ = y_mean - x_mean @ self.coef_    # Step 9
        self.n_features_in_ = p
        return self

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_
