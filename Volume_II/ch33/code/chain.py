# Hierarchical labels: "beach" can only be true when "outdoor" is.
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.multioutput import ClassifierChain as Chain
from sklearn.multioutput import MultiOutputClassifier


def make(n, rng):
    X = rng.uniform(-2, 2, (n, 2))
    outdoor = X[:, 0] > 0
    beach = outdoor & (X[:, 1] > 0)
    return X, np.c_[outdoor, beach].astype(int)


base = LogisticRegression(C=1e4, max_iter=1000)
models = {"independent": MultiOutputClassifier(base),
          "chain outdoor, beach": Chain(base, order=[0, 1]),
          "chain beach, outdoor": Chain(base, order=[1, 0]),
          "chain with cv=5": Chain(base, order=[0, 1], cv=5)}
acc = {k: [] for k in models}
for seed in range(20):                      # 20 random data sets
    rng = np.random.default_rng(seed)
    Xtr, Ytr = make(300, rng)
    Xte, Yte = make(2000, rng)
    for k, m in models.items():
        P = m.fit(Xtr, Ytr).predict(Xte)
        acc[k].append([(P[:, 0] == Yte[:, 0]).mean(),
                       (P[:, 1] == Yte[:, 1]).mean(),
                       accuracy_score(Yte, P)])
print(f"{'':22s}{'outdoor':>9s}{'beach':>9s}{'subset':>9s}")
for k, v in acc.items():
    m = np.mean(v, axis=0)
    print(f"{k:22s}{m[0]:9.4f}{m[1]:9.4f}{m[2]:9.4f}")
link2 = models["chain outdoor, beach"].estimators_[1]
print("link 2 weights on x1, x2, outdoor:", link2.coef_.round(1))
