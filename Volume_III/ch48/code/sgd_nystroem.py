# A kernel one-class SVM and its linear approximation (Chapter 45).
import time
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDOneClassSVM
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score
from sensors import make_sensors

X, _ = make_sensors(n=40000, n_fault=2000, seed=1)
Xt, yt = make_sensors(n=2000, n_fault=100, seed=2)
sc = StandardScaler().fit(X)
Z, Zt = sc.transform(X), sc.transform(Xt)
g = 1 / (Z.shape[1] * Z.var())                # gamma = "scale"
models = {
    "OneClassSVM": OneClassSVM(nu=0.05, gamma=g),
    "Nystroem+SGD": make_pipeline(
        Nystroem(gamma=g, n_components=100, random_state=0),
        SGDOneClassSVM(nu=0.05, tol=None, max_iter=30,
                       random_state=0))}
lab = {}
for name, m in models.items():
    t0 = time.perf_counter()
    m.fit(Z)
    secs = time.perf_counter() - t0
    lab[name] = m.predict(Zt)
    auc = roc_auc_score(yt, -m.score_samples(Zt))
    print(f"{name:13s} fit {secs:5.2f} s  flagged "
          f"{(lab[name] == -1).mean():.3f}  AUC {auc:.3f}")
print("labels agree on",
      (lab["OneClassSVM"] == lab["Nystroem+SGD"]).mean())
