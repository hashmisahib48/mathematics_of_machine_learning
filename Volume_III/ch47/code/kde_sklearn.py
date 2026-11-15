# KernelDensity on the four waiting times of Section 47.1.
import numpy as np
from sklearn.neighbors import KernelDensity

x = np.array([[52.0], [56.0], [80.0], [84.0]])   # shape (n, p)
kde = KernelDensity(kernel="gaussian", bandwidth=4.0).fit(x)
log_f = kde.score_samples([[58.0]])              # ln f(58)
print("ln f(58) =", log_f.round(4), " f(58) =",
      np.exp(log_f).round(5))
print("score(x) =", kde.score(x).round(4), " (a sum, not a mean)")
g = np.linspace(20, 120, 5001)[:, None]
area = np.trapezoid(np.exp(kde.score_samples(g)), g[:, 0])
print("area under f:", area.round(4))
for rule in ["scott", "silverman"]:              # factors only
    k = KernelDensity(bandwidth=rule).fit(x)
    print(f"{rule:9s} bandwidth_ = {k.bandwidth_:.4f}")
