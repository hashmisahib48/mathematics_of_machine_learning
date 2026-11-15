# Exact RBF SVC against an approximate map + SGD, for growing n.
import json
import time
import numpy as np
from sklearn.datasets import make_classification
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = make_classification(n_samples=133000, n_features=20,
                           n_informative=8, flip_y=0.05,
                           class_sep=0.8, random_state=0)
sc = StandardScaler().fit(X[:128000])
Z, Zte = sc.transform(X[:128000]), sc.transform(X[128000:])
yte = y[128000:]
g = 1 / 20                                  # gamma="scale"


def timed(model, n):
    t0 = time.perf_counter()
    model.fit(Z[:n], y[:n])
    return time.perf_counter() - t0, model.score(Zte, yte)


def sgd(n):                                 # alpha = 1/(n C), C = 1
    return SGDClassifier(loss="hinge", alpha=1 / n, max_iter=20,
                         tol=None, random_state=0)


R = {}
print("      n |  exact SVC      | Nystroem(300)+SGD"
      " | RBFSampler(1000)+SGD")
for n in (2000, 8000, 32000, 128000):
    row = []
    if n <= 32000:
        row.append(timed(SVC(gamma=g), n))
    else:
        row.append((float("nan"), float("nan")))
    nys = Nystroem(gamma=g, n_components=300, random_state=0)
    row.append(timed(make_pipeline(nys, sgd(n)), n))
    rff = RBFSampler(gamma=g, n_components=1000, random_state=0)
    row.append(timed(make_pipeline(rff, sgd(n)), n))
    R[n] = row
    print(f"{n:7d} | " + " | ".join(
        f"{t:7.2f} s {a:.4f}" if t == t else f"{'not run':>16s}"
        for t, a in row))
json.dump({str(k): v for k, v in R.items()},
          open("verify/timing.json", "w"))
