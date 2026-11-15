# Section 55.6: a StratifiedKFold from scratch (Eq. 55.3-2).
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold


class MyStratifiedKFold:
    def __init__(self, n_splits=5, shuffle=False, seed=None):
        self.n_splits, self.shuffle = n_splits, shuffle
        self.seed = seed

    def split(self, X, y):
        K = self.n_splits
        _, first, inv = np.unique(y, return_index=True,
                                  return_inverse=True)
        # classes numbered in order of first appearance
        code = np.unique(first, return_inverse=True)[1][inv]
        order = np.sort(code)                  # class blocks
        rng = np.random.RandomState(self.seed)
        fold = np.empty(len(y), dtype=int)
        for c in range(code.max() + 1):
            # a[k] = size of class c in fold k, Eq. 55.3-2
            a = [np.sum(order[k::K] == c) for k in range(K)]
            f = np.repeat(np.arange(K), a)     # fold labels
            if self.shuffle:
                rng.shuffle(f)
            fold[code == c] = f
        for k in range(K):
            test = fold == k
            yield np.flatnonzero(~test), np.flatnonzero(test)


cases = [("cancer", *load_breast_cancer(return_X_y=True)),
         ("12 rows", np.zeros((12, 1)), np.r_[[1] * 7, [0] * 5])]
for name, X, y in cases:
    for sh in (False, True):
        rs = 3 if sh else None
        mine = MyStratifiedKFold(5, sh, rs).split(X, y)
        ref = StratifiedKFold(5, shuffle=sh, random_state=rs)
        for (_, a), (_, b) in zip(mine, ref.split(X, y)):
            assert (a == b).all()
        print(name, "shuffle =", sh, ": folds agree")
