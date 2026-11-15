# SGD for a linear classifier, written from the equations of §17.5.
import numpy as np

SLOPE = {                      # dL/dz, z = y f(x); see §17.4
    "hinge": lambda z: -float(z <= 1),
    "perceptron": lambda z: -float(z <= 0),
    "log_loss": lambda z: -1 / (1 + np.exp(z)),
}

def rate(kind, t, a, eta0, pw):
    if kind == "constant":
        return eta0                          # Eq. 17.5-4
    if kind == "invscaling":
        return eta0 / t ** pw                # Eq. 17.5-5
    return 1 / (a ** 0.25 + a * (t - 1))     # Eq. 17.5-7

class MySGD:
    def __init__(self, loss="hinge", alpha=1e-4, schedule="optimal",
                 eta0=0.01, power_t=0.5, epochs=5, shuffle=True,
                 seed=0):
        self.loss, self.alpha, self.schedule = loss, alpha, schedule
        self.eta0, self.power_t = eta0, power_t
        self.epochs, self.shuffle, self.seed = epochs, shuffle, seed

    def fit(self, X, y):
        X = np.asarray(X, float)
        self.classes_ = np.unique(y)
        s = np.where(y == self.classes_[1], 1.0, -1.0)  # y = +-1
        n, w, b, t = len(X), np.zeros(X.shape[1]), 0.0, 1
        a, rng = self.alpha, np.random.RandomState(self.seed)
        for _ in range(self.epochs):
            order = rng.permutation(n) if self.shuffle else range(n)
            for i in order:
                eta = rate(self.schedule, t, a, self.eta0,
                           self.power_t)
                z = s[i] * (X[i] @ w + b)            # Eq. 17.3-2
                g = s[i] * SLOPE[self.loss](z)       # dL/df
                w = (1 - eta * a) * w - eta * g * X[i]  # Eq. 17.5-3
                b, t = b - eta * g, t + 1
        self.coef_, self.intercept_, self.t_ = w, b, t
        return self

    def predict(self, X):
        f = np.asarray(X, float) @ self.coef_ + self.intercept_
        return self.classes_[(f > 0) * 1]
