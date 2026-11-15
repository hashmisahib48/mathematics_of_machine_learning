# Two silent failures of the graph methods on the digits (34.8).
import warnings
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import LabelPropagation

X, y = load_digits(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)
lab, _ = train_test_split(np.arange(len(ytr)), train_size=30,
                          random_state=0, stratify=ytr)
yy = -np.ones(len(ytr), int)
yy[lab] = ytr[lab]                      # 30 labels, the rest -1
U = yy == -1

D2 = pairwise_distances(Xtr, squared=True)
np.fill_diagonal(D2, np.inf)
nn2 = np.median(D2.min(axis=1))         # typical squared distance
print(f"median squared nearest-neighbour distance: {nn2:.0f}")
print(f"at gamma=20 that weight is exp({-20 * nn2:.0f}) =",
      np.exp(-20 * nn2))


def fit(**kw):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m = LabelPropagation(max_iter=3000, **kw).fit(Xtr, yy)
        m.predict(Xte[:5])
    return m, [x.category.__name__ for x in w]


for g in (20, 0.02):
    m, w = fit(gamma=g)
    acc = np.mean(m.transduction_[U] == ytr[U])
    print(f"rbf gamma={g:<5} n_iter_={m.n_iter_:<5d} accuracy "
          f"{acc:.3f}  warnings: {w}")
for k in (4, 7, 12):
    m, w = fit(kernel="knn", n_neighbors=k)
    zero = m.label_distributions_.sum(axis=1) == 0
    acc = np.mean(m.transduction_[U] == ytr[U])
    print(f"knn k={k:<3d} rows without label mass: {zero.sum():4d}"
          f"  accuracy {acc:.3f}")
