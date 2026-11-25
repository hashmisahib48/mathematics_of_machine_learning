# Linear regression by the normal equations, in plain Python.
class LinearRegression:
    """Ordinary least squares, with the names used by scikit-learn."""

    def __init__(self, fit_intercept=True):
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
        c = [sum(r[j] * yi for r, yi in zip(Xc, yc))     # Step 3
             for j in range(p)]
        w = _solve(A, c)                                 # Steps 4-8
        self.coef_ = w
        b = y_mean - sum(m * v for m, v in zip(x_mean, w))  # Step 9
        self.intercept_ = b
        self.n_features_in_ = p
        return self

    def predict(self, X):
        b, w = self.intercept_, self.coef_       # Alg. 3.2, Step 1
        return [sum(xj * wj for xj, wj in zip(row, w)) + b for row in X]

    def score(self, X, y):
        y_hat, y_mean = self.predict(X), sum(y) / len(y)
        ss_res = sum((yi - fi) ** 2 for yi, fi in zip(y, y_hat))
        ss_tot = sum((yi - y_mean) ** 2 for yi in y)
        return 1 - ss_res / ss_tot


def _solve(A, c, eps=1e-12):
    """Solve A w = c by Gaussian elimination with partial pivoting."""
    p = len(c)
    A, c = [row[:] for row in A], c[:]
    tol = eps * max(abs(A[i][i]) for i in range(p))   # scale of A
    for k in range(p):                                # Step 4
        r = max(range(k, p), key=lambda i: abs(A[i][k]))  # Step 5
        A[k], A[r], c[k], c[r] = A[r], A[k], c[r], c[k]
        if abs(A[k][k]) <= tol:                       # Step 6
            raise ValueError("the columns of X are linearly dependent")
        for i in range(k + 1, p):                     # Step 7
            m = A[i][k] / A[k][k]
            for j in range(k, p):
                A[i][j] -= m * A[k][j]
            c[i] -= m * c[k]
    w = [0.0] * p                                     # Step 8
    for k in range(p - 1, -1, -1):
        s = sum(A[k][j] * w[j] for j in range(k + 1, p))
        w[k] = (c[k] - s) / A[k][k]
    return w
