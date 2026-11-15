# Problem 15B.1: MyPoly with interaction_only, tested against sklearn.
import numpy as np
from itertools import combinations_with_replacement as cwr
from sklearn.preprocessing import PolynomialFeatures

class MyPoly:                   # monomials of total degree <= d
    def __init__(self, degree=2, interaction_only=False):
        self.degree = degree
        self.interaction_only = interaction_only

    def fit(self, X, y=None):
        self.p_ = np.asarray(X).shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        cols = []
        for k in range(self.degree + 1):
            for c in cwr(range(self.p_), k):      # Eq. 15.3-1
                if self.interaction_only and len(set(c)) < len(c):
                    continue                      # a feature repeats
                cols.append(np.prod(X[:, list(c)], axis=1))
        return np.column_stack(cols)

X = np.random.default_rng(0).normal(size=(6, 4))
mine = MyPoly(3, interaction_only=True).fit(X).transform(X)
ref = PolynomialFeatures(3, interaction_only=True).fit_transform(X)
print("shapes:", mine.shape, ref.shape, " agree:",
      mine.shape == ref.shape and np.allclose(mine, ref))
for k in range(4):              # columns by size of the combination
    n_all = len(list(cwr(range(4), k)))
    n_k = sum(len(set(c)) == k for c in cwr(range(4), k))
    print(f"  size {k}: {n_all} products, {n_k} without repeats")
full = MyPoly(3).fit(X).transform(X)
print("without the restriction:", full.shape[1], "columns")
