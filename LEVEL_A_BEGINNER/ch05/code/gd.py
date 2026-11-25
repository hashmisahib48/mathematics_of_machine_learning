# Linear regression by batch and stochastic gradient descent.
import random


class LinearRegressionGD:
    """Minimize the mean squared error by gradient descent."""

    def __init__(self, eta=0.1, n_iter=1000, tol=1e-8):
        self.eta, self.n_iter, self.tol = eta, n_iter, tol

    def fit(self, X, y):
        n, p = len(X), len(X[0])
        xm = [sum(r[j] for r in X) / n for j in range(p)]     # Step 1
        ym = sum(y) / n
        Xc = [[r[j] - xm[j] for j in range(p)] for r in X]
        yc = [v - ym for v in y]
        w, self.cost_ = [0.0] * p, []                         # Step 2
        for k in range(self.n_iter):                          # Step 3
            e = [yi - sum(a * b for a, b in zip(r, w))        # Step 4
                 for r, yi in zip(Xc, yc)]
            self.cost_.append(sum(v * v for v in e) / n)
            g = [-2 / n * sum(r[j] * ei for r, ei in zip(Xc, e))
                 for j in range(p)]                           # Step 5
            w = [wj - self.eta * gj for wj, gj in zip(w, g)]  # Step 6
            if sum(gj * gj for gj in g) ** 0.5 < self.tol:    # Step 7
                break
        self.n_iter_ = k + 1
        self.coef_ = w
        s = sum(m * v for m, v in zip(xm, w))
        self.intercept_ = ym - s                              # Step 8
        return self

    def predict(self, X):
        return [sum(a * b for a, b in zip(r, self.coef_))
                + self.intercept_ for r in X]


def sgd(X, y, eta0=0.05, epochs=20, seed=0):
    """Stochastic gradient descent: one observation per step."""
    n, p = len(X), len(X[0])
    xm = [sum(r[j] for r in X) / n for j in range(p)]
    ym = sum(y) / n
    w, rng = [0.0] * p, random.Random(seed)
    order = list(range(n))
    for t in range(epochs):
        rng.shuffle(order)                     # a new order each epoch
        eta = eta0 / (1 + t)                   # decreasing step
        for i in order:
            xi = [X[i][j] - xm[j] for j in range(p)]
            e = (y[i] - ym) - sum(a * b for a, b in zip(xi, w))
            w = [wj + eta * 2 * a * e for wj, a in zip(w, xi)]
    return w, ym - sum(m * v for m, v in zip(xm, w))
