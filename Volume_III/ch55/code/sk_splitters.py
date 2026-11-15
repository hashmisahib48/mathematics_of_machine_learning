# Section 55.7: every splitter on 12 samples, 4 groups, 2 classes.
import numpy as np
import sklearn.model_selection as ms

X = np.arange(12).reshape(-1, 1)
y = np.array([0, 1] * 6)
g = np.repeat([0, 1, 2, 3], 3)
# (splitter, needs y, needs groups)
rows = [(ms.KFold(4), 0, 0), (ms.StratifiedKFold(3), 1, 0),
    (ms.RepeatedKFold(n_splits=3, n_repeats=2, random_state=0),
     0, 0),
    (ms.ShuffleSplit(5, test_size=3, random_state=0), 0, 0),
    (ms.StratifiedShuffleSplit(5, test_size=4, random_state=0),
     1, 0),
    (ms.LeaveOneOut(), 0, 0), (ms.LeavePOut(2), 0, 0),
    (ms.GroupKFold(4), 0, 1), (ms.StratifiedGroupKFold(2), 1, 1),
    (ms.LeaveOneGroupOut(), 0, 1), (ms.LeavePGroupsOut(2), 0, 1),
    (ms.GroupShuffleSplit(3, test_size=1, random_state=0), 0, 1),
    (ms.TimeSeriesSplit(3), 0, 0),
    (ms.PredefinedSplit([0] * 3 + [1] * 3 + [-1] * 6), 0, 0)]
print(f"{'splitter':24s}{'y':>2s}{'g':>2s}{'splits':>7s}  sizes")
for cv, uy, ug in rows:
    args = (X, y if uy else None, g if ug else None)
    sizes = [len(te) for _, te in cv.split(*args)]
    assert len(sizes) == cv.get_n_splits(*args)
    print(f"{type(cv).__name__:24s}{'y' if uy else '-':>2s}"
          f"{'g' if ug else '-':>2s}{len(sizes):7d}  {sizes[:6]}")
