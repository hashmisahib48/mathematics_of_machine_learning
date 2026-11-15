# k-NN on the fruit data with scikit-learn.
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

X = np.array([[5, 3], [6, 2], [3, 4], [4, 1],
              [7, 5], [8, 8], [6, 9], [9, 8]], dtype=float)
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])      # 0 apricot, 1 plum
q = [[6.0, 5.0]]

for w in ("uniform", "distance"):
    m = KNeighborsClassifier(n_neighbors=3, weights=w).fit(X, y)
    print(f"{w:8s} predict {m.predict(q)}  proba "
          f"{np.round(m.predict_proba(q)[0], 4)}")
dist, idx = m.kneighbors(q)
print("kneighbors: d =", np.round(dist[0], 4), " rows", idx[0] + 1)
print("classes_", m.classes_, " n_samples_fit_", m.n_samples_fit_,
      " metric", m.effective_metric_)

m2 = KNeighborsClassifier(n_neighbors=2).fit(X, y)       # a tie
print("k = 2 tie:", m2.predict(q), m2.predict_proba(q)[0])
m5 = KNeighborsClassifier(5, weights="distance").fit(X, y)
print("query on point 5:", m5.predict_proba([X[4]])[0])  # d = 0

m1 = KNeighborsClassifier(n_neighbors=1).fit(X, y)
print("1-NN training accuracy", m1.score(X, y),
      "  leave-one-out", (m1.predict(None) == y).mean())
