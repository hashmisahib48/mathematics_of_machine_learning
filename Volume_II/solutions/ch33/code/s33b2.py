# Problem 33B.2: where the tie rules of SVC and the wrapper differ
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC

X, y = make_blobs(n_samples=90, centers=3, cluster_std=2.5,
                  random_state=1)
svc = SVC(kernel="linear").fit(X, y)
wrap = OneVsOneClassifier(SVC(kernel="linear")).fit(X, y)
bt = SVC(kernel="linear", break_ties=True).fit(X, y)

lo, hi = X.min(0) - 5, X.max(0) + 5            # enlarged bounding box
G = np.random.default_rng(0).uniform(lo, hi, (200000, 2))
a, b, c = svc.predict(G), wrap.predict(G), bt.predict(G)
print("SVC against wrapper      :", (a != b).sum(), "of", len(G))
print("wrapper against break_ties:", (b != c).sum())
print("SVC against break_ties    :", (a != c).sum())
k = np.flatnonzero(a != b)[0]
g = G[k:k + 1]
print("first point:", g[0].round(3), " SVC", a[k], " wrapper", b[k])
print("pair scores of the wrapper (0,1),(0,2),(1,2):",
      [round(float(e.decision_function(g)[0]), 3)
       for e in wrap.estimators_])
print("wrapper F (33.3-7):", wrap.decision_function(g)[0].round(4))
