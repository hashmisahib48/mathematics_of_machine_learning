# Gradient boosting from scratch: squared error and log loss.
import numpy as np
from scipy.special import expit
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import check_random_state


class MyGBR:
    """Squared-error boosting: trees are fitted to residuals."""

    def __init__(self, n_estimators=100, learning_rate=0.1,
                 max_depth=3, random_state=0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state

    def fit(self, X, y):
        rng = check_random_state(self.random_state)
        self.f0_ = y.mean()                      # best constant
        F = np.full(len(y), self.f0_)
        self.trees_ = []
        for m in range(self.n_estimators):
            r = y - F                            # Eq. 27.4-1
            tree = DecisionTreeRegressor(max_depth=self.max_depth,
                                         random_state=rng).fit(X, r)
            F += self.learning_rate * tree.predict(X)   # Eq. 27.3-6
            self.trees_.append(tree)
        return self

    def predict(self, X):
        F = np.full(len(X), self.f0_)
        for tree in self.trees_:
            F += self.learning_rate * tree.predict(X)
        return F


class MyGBC:
    """Binary log-loss boosting with one Newton step per leaf."""

    def __init__(self, n_estimators=100, learning_rate=0.1,
                 max_depth=3, random_state=0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state

    def fit(self, X, y):
        rng = check_random_state(self.random_state)
        pi = y.mean()
        self.f0_ = np.log(pi / (1 - pi))         # Eq. 27.5-4
        F = np.full(len(y), self.f0_)
        self.trees_, self.leaf_ = [], []
        for m in range(self.n_estimators):
            p = expit(F)
            r = y - p                            # Eq. 27.5-3
            tree = DecisionTreeRegressor(max_depth=self.max_depth,
                                         random_state=rng).fit(X, r)
            leaf = tree.apply(X)
            gamma = np.zeros(tree.tree_.node_count)
            for t in np.unique(leaf):            # Eq. 27.5-6
                k = leaf == t
                gamma[t] = r[k].sum() / (p[k] * (1 - p[k])).sum()
            F += self.learning_rate * gamma[leaf]
            self.trees_.append(tree)
            self.leaf_.append(gamma)
        return self

    def decision_function(self, X):
        F = np.full(len(X), self.f0_)
        for tree, gamma in zip(self.trees_, self.leaf_):
            F += self.learning_rate * gamma[tree.apply(X)]
        return F

    def predict_proba(self, X):
        p = expit(self.decision_function(X))
        return np.c_[1 - p, p]

    def predict(self, X):
        return (self.decision_function(X) >= 0).astype(int)
