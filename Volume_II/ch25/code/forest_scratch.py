# A random forest from scratch: bootstrap, trees, average (Ch. 25).
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

MAX_INT = np.iinfo(np.int32).max


class MyForest:
    def __init__(self, B=100, max_features="sqrt", random_state=0):
        self.B, self.m, self.seed = B, max_features, random_state

    def fit(self, X, y):
        n = len(y)
        self.classes_ = np.unique(y)
        rs = np.random.RandomState(self.seed)
        seeds = rs.randint(MAX_INT, size=self.B)
        self.trees_, self.counts_ = [], []
        for s in seeds:
            idx = np.random.RandomState(s).randint(0, n, n)
            w = np.bincount(idx, minlength=n)      # Eq. 25.3-1
            tree = DecisionTreeClassifier(max_features=self.m,
                                          random_state=s)
            self.trees_.append(tree.fit(X, y, sample_weight=w))
            self.counts_.append(w)
        P, k = np.zeros((n, len(self.classes_))), np.zeros(n)
        for tree, w in zip(self.trees_, self.counts_):
            out = w == 0                 # out of bag, Eq. 25.5-2
            P[out] += tree.predict_proba(X[out])
            k[out] += 1
        self.oob_proba_ = P / k[:, None]
        return self

    def predict_proba(self, X):                    # Eq. 25.3-9
        return np.mean([t.predict_proba(X) for t in self.trees_],
                       axis=0)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)
mine = MyForest(100).fit(X_tr, y_tr)
ref = RandomForestClassifier(n_estimators=100, oob_score=True,
                             random_state=0).fit(X_tr, y_tr)
p_mine, p_ref = mine.predict_proba(X_te), ref.predict_proba(X_te)
print("largest |difference| in P(class):",
      np.abs(p_mine - p_ref).max())
print("same predictions:",
      np.array_equal(mine.predict(X_te), ref.predict(X_te)))
print("same bootstrap samples:",
      all(np.array_equal(w, np.bincount(s, minlength=len(y_tr)))
          for w, s in zip(mine.counts_, ref.estimators_samples_)))
print("largest OOB difference:",
      np.abs(mine.oob_proba_ - ref.oob_decision_function_).max())
oob = (mine.oob_proba_.argmax(axis=1) == y_tr).mean()
print("OOB accuracy:", round(oob, 4), " scikit-learn:",
      round(ref.oob_score_, 4))
acc = (mine.predict(X_te) == y_te).mean()
print("test accuracy:", round(acc, 4))
assert np.allclose(p_mine, p_ref, atol=1e-12)
assert np.allclose(mine.oob_proba_, ref.oob_decision_function_)
