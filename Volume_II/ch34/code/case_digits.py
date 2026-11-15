import warnings
import numpy as np
from sklearn.datasets import load_digits
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import (LabelPropagation as LP,
    LabelSpreading as LS, SelfTrainingClassifier as Self)
warnings.simplefilter("ignore", ConvergenceWarning)
warnings.simplefilter("ignore", RuntimeWarning)       # see 34.8

X, y = load_digits(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)
lr = lambda: make_pipeline(StandardScaler(),
                           LogisticRegression(max_iter=1000))
models = {                       # name: (model, uses unlabelled)
 "logistic, labelled only": (lr(), False),
 "1-NN, labelled only": (KNeighborsClassifier(1), False),
 "LabelPropagation knn 7": (LP(kernel="knn", max_iter=1000), True),
 "LabelSpreading knn 7": (LS(kernel="knn", alpha=0.8), True),
 "LabelSpreading rbf 0.02": (LS(gamma=0.02, alpha=0.8), True),
 "SelfTraining thr 0.75": (Self(lr()), True),
 "SelfTraining k_best 50": (Self(lr(), criterion="k_best",
                                 k_best=50, max_iter=None), True),
}
res = {k: [] for k in models}
for seed in range(5):                    # five draws of 30 labels
    lab, _ = train_test_split(np.arange(len(ytr)), train_size=30,
                              random_state=seed, stratify=ytr)
    yy = -np.ones(len(ytr), int)
    yy[lab] = ytr[lab]
    U = yy == -1
    for k, (m, semi) in models.items():
        m.fit(Xtr, yy) if semi else m.fit(Xtr[lab], ytr[lab])
        a = np.mean(m.predict(Xtr[U]) == ytr[U])
        res[k].append([a, m.score(Xte, yte)])
print(f"{'model':25s} unlabelled        test")
for k, v in res.items():
    v = np.array(v)
    print(f"{k:25s} {v[:, 0].mean():.3f}+-{v[:, 0].std():.3f}"
          f"  {v[:, 1].mean():.3f}+-{v[:, 1].std():.3f}")
