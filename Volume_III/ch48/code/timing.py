# Fit time of the exact one-class SVM and of Nystroem + SGD against n.
import json, time
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDOneClassSVM
from sklearn.svm import OneClassSVM
from sensors import make_sensors

Xt, yt = make_sensors(n=2000, n_fault=100, seed=2)
T = {}
for n in (2000, 5000, 10000, 20000, 40000, 80000, 160000):
    X, _ = make_sensors(n=n, n_fault=n // 20, seed=1)
    sc = StandardScaler().fit(X)
    Z = sc.transform(X)
    g = 1 / (Z.shape[1] * Z.var())
    row = []
    for name, m in (("exact", OneClassSVM(nu=0.05, gamma=g)),
                    ("approx", make_pipeline(
                        Nystroem(gamma=g, n_components=100,
                                 random_state=0),
                        SGDOneClassSVM(nu=0.05, tol=None, max_iter=30,
                                       random_state=0)))):
        if name == "exact" and n > 40000:
            row.append(None)           # minutes: not run
            continue
        t0 = time.perf_counter()
        m.fit(Z)
        row.append(round(time.perf_counter() - t0, 3))
    T[n] = row
    print(n, row)
json.dump(T, open("verify/timing.json", "w"))
