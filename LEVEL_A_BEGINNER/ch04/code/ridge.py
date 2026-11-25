# Polynomial features and ridge regression, in plain Python.
from linreg import _solve


def poly_features(X, degree):
    """Powers 1..degree of every feature (no cross products)."""
    return [[v ** d for v in row for d in range(1, degree + 1)]
            for row in X]


class Ridge:
    """Least squares with the penalty alpha * sum(w_j ** 2)."""

    def __init__(self, alpha=1.0, fit_intercept=True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        n, p = len(X), len(X[0])
        if self.fit_intercept:                                  # Step 1
            x_mean = [sum(row[j] for row in X) / n for j in range(p)]
            y_mean = sum(y) / n
        else:
            x_mean, y_mean = [0.0] * p, 0.0
        Xc = [[row[j] - x_mean[j] for j in range(p)] for row in X]
        yc = [yi - y_mean for yi in y]
        A = [[sum(r[j] * r[k] for r in Xc) for k in range(p)]  # Step 2
             for j in range(p)]
        for j in range(p):                                      # Step 3
            A[j][j] += self.alpha
        c = [sum(r[j] * yi for r, yi in zip(Xc, yc))           # Step 4
             for j in range(p)]
        self.coef_ = _solve(A, c)                               # Step 5
        self.intercept_ = y_mean - sum(m * w for m, w           # Step 6
                                       in zip(x_mean, self.coef_))
        return self

    def predict(self, X):
        return [sum(x * w for x, w in zip(row, self.coef_))
                + self.intercept_ for row in X]
