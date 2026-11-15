# How the fit time grows with the number of samples.
import time
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

ns, ts = [], []
print("     n  SVC rbf (s)   n_SV  LinearSVC (s)  SVC linear (s)")
for n in (1000, 2000, 4000, 8000, 16000, 32000):
    X, y = make_classification(n_samples=n, n_features=20,
                               n_informative=8, flip_y=0.05,
                               class_sep=0.8, random_state=0)
    Z = StandardScaler().fit_transform(X)
    res = []
    for est in (SVC(), LinearSVC(), SVC(kernel="linear")):
        if n > 8000 and est.get_params().get("kernel") == "linear":
            res.append(float("nan"))
            continue
        t0 = time.perf_counter()
        est.fit(Z, y)
        res.append(time.perf_counter() - t0)
        if isinstance(est, SVC) and est.kernel == "rbf":
            nsv = len(est.support_)
    ns.append(n)
    ts.append(res[0])
    print(f"{n:6d} {res[0]:11.2f} {nsv:6d} {res[1]:13.3f}"
          f" {res[2]:14.2f}")
slope = np.polyfit(np.log(ns[1:]), np.log(ts[1:]), 1)[0]
print(f"fit time of SVC rbf grows like n^{slope:.2f} (n >= 2000)")
