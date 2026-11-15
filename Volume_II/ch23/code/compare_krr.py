# SVR against KernelRidge on Friedman #1: accuracy, time, sparsity.
import time
import numpy as np
from sklearn.datasets import make_friedman1
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from threadpoolctl import threadpool_limits

threadpool_limits(1)                # one BLAS thread: steady times


def data(n, seed):
    return make_friedman1(n, noise=1.0, random_state=seed)


def best_time(model, X, y):              # fastest of two fits
    t = []
    for _ in range(2):
        t0 = time.perf_counter()
        model.fit(X, y)
        t.append(time.perf_counter() - t0)
    return min(t)


X, y = data(1000, 0)
X_te, y_te = data(2000, 1)
g = [0.03, 0.1, 0.3]
svr = GridSearchCV(SVR(), {"gamma": g, "C": [1, 10, 100],
                           "epsilon": [0.3, 1, 2]}, cv=3).fit(X, y)
krr = GridSearchCV(KernelRidge(kernel="rbf"),
                   {"gamma": g, "alpha": [0.01, 0.1, 1]}, cv=3)
krr.fit(X, y - y.mean())
print("SVR", svr.best_params_, "\nKRR", krr.best_params_)

print("    n  fit SVR  fit KRR  SVs   R2 SVR  R2 KRR")
for n in (250, 500, 1000, 2000, 4000):
    X, y = data(n, 0)
    s = SVR(**svr.best_params_)
    k = KernelRidge(kernel="rbf", **krr.best_params_)
    t_s, t_k = best_time(s, X, y), best_time(k, X, y - y.mean())
    p = k.predict(X_te) + y.mean()
    r2 = 1 - ((y_te - p)**2).sum() / ((y_te - y_te.mean())**2).sum()
    print(f"{n:5d} {t_s:7.2f}s {t_k:7.2f}s {len(s.support_):5d}"
          f"  {s.score(X_te, y_te):7.3f} {r2:7.3f}")
