# Logistic regression trained by batch gradient descent.
import math


def sigmoid(t):
    return 1.0 / (1.0 + math.exp(-t))


class LogisticRegression:
    def __init__(self, eta=0.1, n_iter=1000, tol=1e-6):
        self.eta, self.n_iter, self.tol = eta, n_iter, tol

    def decision_function(self, X):
        return [self.intercept_ + sum(w * v for w, v in
                                      zip(self.coef_, x)) for x in X]

    def predict_proba(self, X):
        return [sigmoid(t) for t in self.decision_function(X)]

    def predict(self, X):
        return [int(p >= 0.5) for p in self.predict_proba(X)]

    def fit(self, X, y):
        n, p = len(X), len(X[0])
        self.coef_, self.intercept_ = [0.0] * p, 0.0          # Step 1
        self.cost_ = []
        for it in range(self.n_iter):
            pr = self.predict_proba(X)                         # Step 2
            e = [a - b for a, b in zip(pr, y)]                 # Step 3
            gw = [sum(ei * x[j] for ei, x in zip(e, X)) / n
                  for j in range(p)]
            gb = sum(e) / n
            self.cost_.append(-sum(b * math.log(a) + (1 - b) *
                  math.log(1 - a) for a, b in zip(pr, y)) / n)
            self.coef_ = [w - self.eta * g                     # Step 4
                          for w, g in zip(self.coef_, gw)]
            self.intercept_ -= self.eta * gb
            if max(abs(g) for g in gw + [gb]) < self.tol:     # Step 5
                break
        self.n_iter_ = it + 1
        return self

    def score(self, X, y):
        return sum(a == b for a, b in zip(self.predict(X), y)) / len(y)
