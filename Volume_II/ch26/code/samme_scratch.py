# SAMME from scratch, with decision stumps as the weak learner.
import numpy as np
from sklearn.tree import DecisionTreeClassifier


class MyAdaBoost:
    def __init__(self, n_estimators=50, learning_rate=1.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

    def fit(self, X, y):
        self.classes_, yi = np.unique(y, return_inverse=True)
        K, n = len(self.classes_), len(yi)
        w = np.full(n, 1.0 / n)
        self.stumps_, self.alphas_, self.errors_ = [], [], []
        for m in range(self.n_estimators):
            h = DecisionTreeClassifier(max_depth=1, random_state=0)
            h.fit(X, yi, sample_weight=w)
            wrong = h.predict(X) != yi
            err = np.average(wrong, weights=w)      # Eq. 26.3-1
            if err >= 1 - 1 / K:           # no better than a guess
                break
            a = 1.0                         # as scikit-learn
            if err > 0:                             # Eq. 26.3-2
                a = self.learning_rate * (
                    np.log((1 - err) / err) + np.log(K - 1))
            self.stumps_.append(h)
            self.alphas_.append(a)
            self.errors_.append(err)
            if err == 0:
                break
            w = w * np.exp(a * wrong)               # Eq. 26.3-3
            w = w / w.sum()
        return self

    def decision_function(self, X):                 # Eq. 26.3-4
        K = len(self.classes_)
        S = sum(a * (h.predict(X)[:, None] == np.arange(K))
                for h, a in zip(self.stumps_, self.alphas_))
        return S

    def predict(self, X):
        return self.classes_[self.decision_function(X).argmax(1)]
