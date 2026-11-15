# Problem 33B.3: a classifier chain built by hand
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier

def make(n, rng):
    X = rng.uniform(-2, 2, (n, 2))
    y1 = X[:, 0] > 0                       # outdoor
    y2 = y1 & (X[:, 1] > 0)                # beach
    return X, np.c_[y1, y2].astype(int)

rng = np.random.default_rng(0)
Xtr, Ytr = make(300, rng)
Xte, Yte = make(2000, rng)
lr = lambda: LogisticRegression(C=1e4, max_iter=1000)

l1 = lr().fit(Xtr, Ytr[:, 0])                    # link 1: x
l2 = lr().fit(np.c_[Xtr, Ytr[:, 0]], Ytr[:, 1])  # link 2: x, TRUE y1
p1 = l1.predict(Xte)
p2 = l2.predict(np.c_[Xte, p1])                  # PREDICTED y1 now
hand = np.c_[p1, p2]

chain = ClassifierChain(lr(), order=[0, 1]).fit(Xtr, Ytr).predict(Xte)
ind = MultiOutputClassifier(lr()).fit(Xtr, Ytr).predict(Xte)
print("hand-made equals ClassifierChain:", (hand == chain).all())
for name, P in (("chain by hand", hand), ("independent", ind)):
    print(f"{name:14s} outdoor {np.mean(P[:, 0] == Yte[:, 0]):.4f}"
          f"  beach {np.mean(P[:, 1] == Yte[:, 1]):.4f}"
          f"  subset {accuracy_score(Yte, P):.4f}")
print("link 2 weights on x1, x2, y1:", l2.coef_.round(1)[0])
t2 = l2.predict(np.c_[Xte, Yte[:, 0]])      # with the true y1
print("link 2 given the true y1: beach", np.mean(t2 == Yte[:, 1]))
