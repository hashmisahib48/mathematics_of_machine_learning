# Test of MyKNN against scikit-learn.
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from knn_scratch import MyKNN

rng = np.random.default_rng(0)
X = rng.normal(size=(300, 4)) + rng.integers(0, 3, 300)[:, None]
s = X[:, 0] + X[:, 1] + rng.normal(size=300)
y = np.digitize(s, [1, 3])                     # three classes
Xq = np.vstack([rng.normal(size=(100, 4)) + 1.5, X[:3]])
for w in ("uniform", "distance"):
    mine = MyKNN(7, w).fit(X, y)
    ref = KNeighborsClassifier(7, weights=w).fit(X, y)
    assert (mine.predict(Xq) == ref.predict(Xq)).all()
    p_a, p_b = mine.predict_proba(Xq), ref.predict_proba(Xq)
    assert np.allclose(p_a, p_b)
    d_a, i_a = mine.kneighbors(Xq)
    d_b, i_b = ref.kneighbors(Xq)
    assert (i_a == i_b).all() and np.allclose(d_a, d_b)
    print(f"{w:8s}: predict, predict_proba, kneighbors agree;",
          "class counts", np.bincount(mine.predict(Xq)))
