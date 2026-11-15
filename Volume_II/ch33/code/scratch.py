# One-versus-rest and one-versus-one from scratch (Section 33.7).
import numpy as np
from itertools import combinations
from sklearn.base import clone
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier as OvR
from sklearn.multiclass import OneVsOneClassifier as OvO
from sklearn.preprocessing import StandardScaler


class OneVs:
    def __init__(self, base, scheme="ovr"):
        self.base, self.scheme = base, scheme

    def fit(self, X, y):
        self.classes_ = c = np.unique(y)
        if self.scheme == "ovr":                 # K problems
            self.pairs_ = [(k,) for k in range(len(c))]
        else:                                    # K(K-1)/2 problems
            self.pairs_ = list(combinations(range(len(c)), 2))
        self.models_ = []
        for p in self.pairs_:
            m = slice(None)
            if len(p) == 2:
                m = np.isin(y, c[list(p)])
            z = y[m] == c[p[-1]]            # the last class is 1
            self.models_.append(clone(self.base).fit(X[m], z * 1))
        return self

    def decision_function(self, X):
        f = [m.decision_function(X) for m in self.models_]
        if self.scheme == "ovr":
            return np.column_stack(f)            # Eq. 33.3-2
        n, K = len(X), len(self.classes_)
        votes, c = np.zeros((n, K)), np.zeros((n, K))
        for (j, k), fp in zip(self.pairs_, f):
            votes[np.arange(n), np.where(fp > 0, k, j)] += 1
            c[:, j] -= fp                        # Eq. 33.3-6
            c[:, k] += fp
        return votes + c / (3 * (abs(c) + 1))    # Eq. 33.3-7

    def predict(self, X):
        return self.classes_[self.decision_function(X).argmax(1)]


# ---- test against the wrappers
X, y = load_digits(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)
sc = StandardScaler().fit(Xtr)
Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
base = LogisticRegression(max_iter=300)
for s, ref in [("ovr", OvR(base)), ("ovo", OvO(base))]:
    mine = OneVs(base, s).fit(Xtr, ytr)
    ref.fit(Xtr, ytr)
    d = abs(mine.decision_function(Xte)
            - ref.decision_function(Xte))
    assert (mine.predict(Xte) == ref.predict(Xte)).all()
    assert d.max() < 1e-9
    acc = (mine.predict(Xte) == yte).mean()
    print(s, "models:", len(mine.models_),
          " max diff:", f"{d.max():.1e}",
          " accuracy:", round(acc, 4))
