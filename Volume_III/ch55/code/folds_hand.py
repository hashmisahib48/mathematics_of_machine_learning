# Section 55.4: ten samples, by hand and by scikit-learn.
import numpy as np
import sklearn.model_selection as ms

y = np.array([0, 0, 1, 0, 1, 0, 0, 1, 0, 1])    # six 0s, four 1s
X = np.arange(10).reshape(-1, 1)
for name, cv in (("KFold", ms.KFold(3)),
                 ("Stratified", ms.StratifiedKFold(3))):
    print(f"{name:10s}", [t.tolist() for _, t in cv.split(X, y)])
# round robin over the sorted labels (Eq. 55.3-2)
order = np.sort(y)
alloc = [np.bincount(order[k::3], minlength=2) for k in range(3)]
print("allocation", [a.tolist() for a in alloc])
for a, (_, t) in zip(alloc, ms.StratifiedKFold(3).split(X, y)):
    assert a.tolist() == np.bincount(y[t], minlength=2).tolist()
for tr, te in ms.TimeSeriesSplit(3, gap=1).split(X):
    print("gap=1  train", tr.tolist(), " test", te.tolist())
