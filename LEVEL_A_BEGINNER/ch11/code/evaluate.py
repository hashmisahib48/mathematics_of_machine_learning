# Standardization, k-fold cross-validation and classification metrics.
import math
import random
from itertools import product

class StandardScaler:
    def fit(self, X):
        n = len(X)
        self.mean_ = [sum(c) / n for c in zip(*X)]
        self.scale_ = [math.sqrt(sum((v - m) ** 2 for v in c) / n)
                       for c, m in zip(zip(*X), self.mean_)]
        return self

    def transform(self, X):                           # Eq. 11.2-1
        return [[(v - m) / s for v, m, s in
                 zip(x, self.mean_, self.scale_)] for x in X]

def kfold(n, k, seed=0):                              # Algorithm 11.1
    idx = list(range(n))
    random.Random(seed).shuffle(idx)                  # Step 1
    folds = [idx[f::k] for f in range(k)]             # Step 2
    for f in range(k):
        test = folds[f]
        train = [i for g in range(k) if g != f for i in folds[g]]
        yield train, test

def cross_val_score(make_model, X, y, k=5, seed=0):
    scores = []
    for train, test in kfold(len(y), k, seed):        # Step 3
        sc = StandardScaler().fit([X[i] for i in train])
        m = make_model().fit(sc.transform([X[i] for i in train]),
                             [y[i] for i in train])
        pred = m.predict(sc.transform([X[i] for i in test]))
        scores.append(sum(p == y[i] for p, i in zip(pred, test))
                      / len(test))                    # Step 4
    return scores                                     # Step 5

def confusion(y, pred):
    tp = sum(a == 1 and b == 1 for a, b in zip(y, pred))
    fp = sum(a == 0 and b == 1 for a, b in zip(y, pred))
    fn = sum(a == 1 and b == 0 for a, b in zip(y, pred))
    tn = len(y) - tp - fp - fn
    return tp, fp, fn, tn

def precision_recall_f1(y, pred):         # Eqs. 11.2-4, 11.2-5
    tp, fp, fn, tn = confusion(y, pred)
    p, r = tp / (tp + fp), tp / (tp + fn)
    return p, r, 2 * p * r / (p + r)

def roc_auc(y, score):                                # Eq. 11.2-7
    pos = [s for s, t in zip(score, y) if t == 1]
    neg = [s for s, t in zip(score, y) if t == 0]
    wins = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
    return wins / (len(pos) * len(neg))

def grid_search(make_model, X, y, grid, k=5):         # Algorithm 11.2
    scored = []
    for values in product(*grid.values()):            # Step 1
        p = dict(zip(grid, values))
        s = cross_val_score(lambda: make_model(**p), X, y, k)
        scored.append((sum(s) / k, p))                # Step 2
    return max(scored, key=lambda t: t[0])            # Step 3
