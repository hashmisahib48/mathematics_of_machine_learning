# One-feature k-NN regression on the delivery data (Chapter 9).
import numpy as np

x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])     # distance, km
y = np.array([3.0, 4.0, 6.0, 7.0, 10.0])    # time, min

def knn_predict(x0, k, weighted):
    d = np.abs(x - x0)                       # 1-D distance
    nn = np.argsort(d, kind="stable")[:k]    # the k nearest
    w = 1 / d[nn] if weighted else np.ones(k)   # Eq. 20.3-5
    return (w * y[nn]).sum() / w.sum(), nn + 1, d[nn]  # Eq. 20.3-7

for x0 in (3.6, 6.0):
    for k, weighted in ((2, False), (2, True), (3, False)):
        p, nn, dd = knn_predict(x0, k, weighted)
        kind = "1/d " if weighted else "mean"
        print(f"x0={x0}  k={k} {kind} neighbours {nn} "
              f"d={np.round(dd, 1)}  prediction {p:.4f}")
b, w = 0.9, 1.7                      # Example 9.5-1 (Volume I)
print(f"least squares: {b + w * 3.6:.2f} and {b + w * 6.0:.2f}")
