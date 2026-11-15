# Problem 34B.2: transduction_ against predict on the digits.
import warnings
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
warnings.simplefilter("ignore")      # convergence, divide warnings
X, y = load_digits(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)
lab, _ = train_test_split(np.arange(len(ytr)), train_size=30,
                          random_state=0, stratify=ytr)
yy = -np.ones(len(ytr), int)
yy[lab] = ytr[lab]                   # 30 labels, -1 elsewhere
unl = yy == -1
nbr = NearestNeighbors(n_neighbors=7).fit(Xtr).kneighbors(
    Xtr, return_distance=False)      # itself is the first
models = {"propagation": LabelPropagation(
              kernel="knn", n_neighbors=7, max_iter=1000),
          "spreading": LabelSpreading(
              kernel="knn", n_neighbors=7, alpha=0.8)}
for name, m in models.items():
    m.fit(Xtr, yy)
    tr, pr = m.transduction_, m.predict(Xtr)
    d = pr != tr
    print(f"{name}: {d.sum()} differ, {(d & ~unl).sum()} "
          f"labelled, {(d & unl).sum()} unlabelled")
    print(f"  accuracy, unlabelled: transduction_ "
          f"{np.mean(tr[unl] == ytr[unl]):.4f}, "
          f"predict {np.mean(pr[unl] == ytr[unl]):.4f}")
    for i in np.where(d & ~unl)[0]:  # labelled, outvoted
        v = np.bincount(tr[nbr[i]], minlength=10)
        print(f"  sample {i}: label {ytr[i]}, predict {pr[i]}, "
              f"votes {v[ytr[i]]} for the label, {v[pr[i]]} for it")
    c = d & unl                      # unlabelled, answers differ
    if c.any():
        print(f"  of {c.sum()} unlabelled: predict right at "
              f"{(pr[c] == ytr[c]).sum()}, transduction_ at "
              f"{(tr[c] == ytr[c]).sum()}")
